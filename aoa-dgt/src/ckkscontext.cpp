#include <AOADGT/ckkscontext.h>
#include <AOADGT/tool/encoder.h>

void CKKSContext::encode(poly_t *a, int64_t *scale, complex<double>* val, int slots, int empty_slots) {
	Logger::getInstance()->log_debug("CKKSContext::encode");
	assert(a->state != NONINITIALIZED);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	// Encode
	if(*scale == -1)
		// *scale = COPRIMES_BUCKET[1];
		*scale = ((uint64_t)1 << CUDAEngine::scalingfactor);
	ckks_encode(
		this,
		a->d_coefs,
		val,
		slots,
		empty_slots,
		*scale);

	cudaStreamSynchronize(this->get_stream());
	cudaCheckError();
	
	//////////////////
	// DGT
	DGTEngine::execute_dgt(this, a, FORWARD);

}

void CKKSContext::decode(complex<double> *val, poly_t *a, int64_t scale, int slots) {
	Logger::getInstance()->log_debug("CKKSContext::decode");
	assert(a->state != NONINITIALIZED);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	//////////////////
	// IDGT
	DGTEngine::execute_dgt(this, a, INVERSE);

	cudaStreamSynchronize(this->get_stream());
	cudaCheckError();

	// Decode
	ckks_decode(this, val, a->d_coefs, slots, scale);

}

void CKKSContext::encodeSingle(poly_t *a, int64_t *scale, complex<double> val) {
	Logger::getInstance()->log_debug("CKKSContext::encodeSingle");
	assert(a->state != NONINITIALIZED);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	//
	memset(this->h_coefs, 0, poly_get_size(a->base));

	// Encode
	if(*scale == -1)
		// *scale = COPRIMES_BUCKET[1];
		*scale = ((uint64_t)1 << CUDAEngine::scalingfactor);
	ckks_encode_single(this->h_coefs, val, *scale);

	//////////////////
	// Copy to device
	cudaMemcpyAsync(
		a->d_coefs,
		this->h_coefs,
		poly_get_size(a->base),
		cudaMemcpyHostToDevice,
		this->get_stream()
	);
	cudaCheckError();

	a->state = RNSSTATE;
	cudaStreamSynchronize(this->get_stream());
	cudaCheckError();

}

void CKKSContext::decodeSingle(complex<double> *val, poly_t *a, int64_t scale) {
	Logger::getInstance()->log_debug("CKKSContext::decodeSingle");

	assert(a->state != NONINITIALIZED);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	//////////////////
	// Copy to Host //
	//////////////////
	DGTEngine::execute_dgt(this, a, INVERSE);

	// Recovers RNS's residues and calls IRNS
	cudaMemcpyAsync(
		this->h_coefs,
		a->d_coefs,
		poly_get_size(a->base),
		cudaMemcpyDeviceToHost,
		this->get_stream()
	);

	cudaStreamSynchronize(this->get_stream());
	cudaCheckError();

	// Decode
	ckks_decode_single(val, this->h_coefs, scale);
}

__host__ void CKKSContext::clear_keys(){

	poly_clear(this, &sk->s);
	poly_clear(this, &pk->a);
	poly_clear(this, &pk->b);
	poly_clear(this, &evk->a);
	poly_clear(this, &evk->b);
}

__host__ void CKKSContext::load_keys(const json & k){

	poly_copy(this, &sk->s, poly_import(this, k["sk"]["s"].GetString()));
	poly_copy(this, &pk->a, poly_import(this, k["pk"]["a"].GetString()));
	poly_copy(this, &pk->b, poly_import(this, k["pk"]["b"].GetString()));
	poly_copy(this, &evk->a, poly_import(this, k["evk"]["a"].GetString()));
	poly_copy(this, &evk->b, poly_import(this, k["evk"]["b"].GetString()));

}

__host__ json CKKSContext::export_keys(){

	std::string sks = poly_export(this, &sk->s);
	std::string pka = poly_export(this, &pk->a);
	std::string pkb = poly_export(this, &pk->b);
	std::string evka = poly_export(this, &evk->a);
	std::string evkb = poly_export(this, &evk->b);

	json jkeys;
	jkeys.SetObject();
	jkeys.AddMember("sk", Value{}.SetObject(), jkeys.GetAllocator());
	jkeys.AddMember("pk", Value{}.SetObject(), jkeys.GetAllocator());
	jkeys.AddMember("evk", Value{}.SetObject(), jkeys.GetAllocator());
	jkeys["sk"].AddMember("s",
		Value{}.SetString(sks.c_str(),
		sks.length(),
		jkeys.GetAllocator()),
		jkeys.GetAllocator());
	jkeys["pk"].AddMember("a",
		Value{}.SetString(pka.c_str(),
		pka.length(),
		jkeys.GetAllocator()),
		jkeys.GetAllocator());
	jkeys["pk"].AddMember("b",
		Value{}.SetString(pkb.c_str(),
		pkb.length(),
		jkeys.GetAllocator()),
		jkeys.GetAllocator());
	jkeys["evk"].AddMember("a",
		Value{}.SetString(evka.c_str(),
		evka.length(),
		jkeys.GetAllocator()),
		jkeys.GetAllocator());
	jkeys["evk"].AddMember("b",
		Value{}.SetString(evkb.c_str(),
		evkb.length(),
		jkeys.GetAllocator()),
		jkeys.GetAllocator());

	return jkeys;
}