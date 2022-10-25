#include <newckks/ckks/ckkscontext.h>
#include <newckks/cuda/manager.h>
#include <newckks/ckks/encoder.h>

void CKKSContext::encode(poly_t *a, int64_t *scale, std::complex<double>* val, int slots, int empty_slots) {
	assert(CUDAManager::is_init);
	poly_init(this, a);

	// Encode
	if(*scale == -1)
		// *scale = COPRIMES_BUCKET[1];
		*scale = ((uint64_t)1 << CUDAManager::scalingfactor);
	ckks_encode(
		this, a->d_coefs, val,
		slots, empty_slots,
		*scale);

	a->state = RNSSTATE;
}

void CKKSContext::decode(std::complex<double> *val, poly_t *a, int64_t scale, int slots) {
	assert(CUDAManager::is_init);
	poly_init(this, a);

	//////////////////
	COMMONEngine::execute(this, a, INVERSE);
	cudaMemcpyAsync(
		this->d_tmp,
		a->d_coefs,
		poly_get_size(a->base),
		cudaMemcpyDeviceToDevice,
		this->get_stream() );
	cudaCheckError();

	// Decode
	ckks_decode(this, val, this->d_tmp, slots, scale);

}


__host__ void CKKSContext::clear_keys(){

	poly_clear(this, &keys.sk->s);
	poly_clear(this, &keys.pk->a);
	poly_clear(this, &keys.pk->b);
	poly_clear(this, &keys.evk->a);
	poly_clear(this, &keys.evk->b);
}

__host__ void CKKSContext::load_keys(const json & k){

	poly_copy(this, &keys.sk->s, poly_import(this, k["sk"]["s"].GetString()));
	poly_copy(this, &keys.pk->a, poly_import(this, k["pk"]["a"].GetString()));
	poly_copy(this, &keys.pk->b, poly_import(this, k["pk"]["b"].GetString()));
	poly_copy(this, &keys.evk->a, poly_import(this, k["evk"]["a"].GetString()));
	poly_copy(this, &keys.evk->b, poly_import(this, k["evk"]["b"].GetString()));

}

__host__ json CKKSContext::export_keys(){

	std::string sks = poly_export(this, &keys.sk->s);
	std::string pka = poly_export(this, &keys.pk->a);
	std::string pkb = poly_export(this, &keys.pk->b);
	std::string evka = poly_export(this, &keys.evk->a);
	std::string evkb = poly_export(this, &keys.evk->b);

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