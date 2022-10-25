#include <AOADGT/arithmetic/polynomial.h>
#include <iterator>

#define hasSupportStreamAlloc()({int r; cudaDeviceGetAttribute(&r, cudaDevAttrMemoryPoolsSupported, 0); r;})

uint64_t get_cycles() {
  unsigned int hi, lo;
  asm (
    "cpuid\n\t"/*serialize*/
    "rdtsc\n\t"/*read the clock*/
    "mov %%edx, %0\n\t"
    "mov %%eax, %1\n\t"
    : "=r" (hi), "=r" (lo):: "%rax", "%rbx", "%rcx", "%rdx"
  );
  return ((uint64_t) lo) | (((uint64_t) hi) << 32);
}

//////////////
// Internal //
//////////////
__host__ int isPowerOfTwo (unsigned int x)
{
  return ((x != 0) && ((x & (~x + 1)) == x));
}

//////////////
//////////////
//////////////

__host__ size_t poly_get_size(poly_bases base){

	return (CUDAEngine::N) * CUDAEngine::get_n_residues(base) *	sizeof(GaussianInteger);
}

__host__  void poly_init(Context *ctx, poly_t *a, poly_bases base){
	if(a->state != NONINITIALIZED)
		return;
	assert(ctx);
	assert(CUDAEngine::is_init);
	Logger::getInstance()->log_debug("poly_init");

	//
	a->base = base; // By default

	// Device
	if(hasSupportStreamAlloc()){
		cudaMallocAsync((void**)&a->d_coefs, poly_get_size(base), ctx->get_stream());
		cudaCheckError();
		cudaMemsetAsync(a->d_coefs, 0, poly_get_size(base), ctx->get_stream());
		cudaCheckError();
	}
	else{
		cudaMalloc((void**)&a->d_coefs, poly_get_size(base));
		cudaCheckError();
		cudaMemset(a->d_coefs, 0, poly_get_size(base));
		cudaCheckError();
	}

	a->state = RNSSTATE;
}

__host__  void poly_free(Context *ctx, poly_t *a){
	if(!a || a->state == NONINITIALIZED)
		return;

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	// RNS residues
	if(hasSupportStreamAlloc())
		cudaFreeAsync(a->d_coefs, ctx->get_stream());
	else{
		cudaStreamSynchronize(ctx->get_stream());
		cudaFree(a->d_coefs);
	}
	cudaCheckError();

	a->d_coefs = NULL;
	a->state = NONINITIALIZED;
}

__host__  void poly_clear(Context *ctx, poly_t *a){
	assert(CUDAEngine::is_init);
	poly_init(ctx, a);
	
	cudaMemsetAsync(a->d_coefs,0, poly_get_size(a->base), ctx->get_stream());
	cudaCheckError();

	a->state = RNSSTATE;
}

__host__ void poly_copy_to_device(Context *ctx, poly_t *a, uint64_t *h_coefs){
	Logger::getInstance()->log_debug("poly_copy_to_device");
	
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	// Computes RNS's residues and copy to the GPU
	memset(ctx->h_coefs, 0, poly_get_size(a->base));
	for(int rid = 0; rid < CUDAEngine::get_n_residues(a->base); rid++)
		for(unsigned int cid = 0; cid < (unsigned int) CUDAEngine::N; cid++){
			// Fold
			ctx->h_coefs[cid + rid * CUDAEngine::N].re   = h_coefs[cid + 2 * rid * CUDAEngine::N];
			ctx->h_coefs[cid + rid * CUDAEngine::N].imag = h_coefs[(cid + CUDAEngine::N) + 2 * rid * CUDAEngine::N];
	}

	cudaMemcpyAsync(
		a->d_coefs,
		ctx->h_coefs,
		poly_get_size(a->base),
		cudaMemcpyHostToDevice,
		ctx->get_stream()
	);
	cudaCheckError();

	a->state = RNSSTATE;
	return;
}

__host__ uint64_t* poly_copy_to_host(Context *ctx, poly_t *a){
	poly_init(ctx, a);

	//////////////////
	// Copy to Host //
	//////////////////
	DGTEngine::execute_dgt( ctx, a, INVERSE);

	// The output
	GaussianInteger *h_coefs = (GaussianInteger*) malloc(poly_get_size(a->base));

	// Recovers RNS's residues and calls IRNS
	cudaMemcpyAsync(
		h_coefs,
		a->d_coefs,
		poly_get_size(a->base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);
	cudaCheckError();

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	uint64_t *result = (uint64_t*) malloc(poly_get_size(a->base));
	// Unfold
	for(int rid = 0; rid < CUDAEngine::get_n_residues(a->base); rid++)
		for(unsigned int cid = 0; cid < (unsigned int) CUDAEngine::N; cid++){
			result[cid 									 + 2 * rid * CUDAEngine::N] = h_coefs[cid + rid * CUDAEngine::N].re;
			result[(cid + CUDAEngine::N) + 2 * rid * CUDAEngine::N] = h_coefs[cid + rid * CUDAEngine::N].imag;
		}

	free(h_coefs);
	return result;
}

__host__ void poly_modup(
	Context *ctx, // Default context
	poly_t *a,  // Output in base QB
	poly_t *b,  // Input in base QB
	int level){
	assert(b->base == QBase);

	poly_init(ctx, a);
	poly_init(ctx, b);

	poly_copy(ctx, a, b);

	DGTEngine::execute_dgt( ctx, a, INVERSE);

	CUDAEngine::execute_approx_modulus_raising(ctx, a->d_coefs, level);

	a->base = QBBase;

}

__host__ void poly_moddown(
	Context *ctx, // Default context
	poly_t *a, // Output in base Q
	poly_t *b, // Input in base QB
	int level){
	assert(b->base == QBBase);

	DGTEngine::execute_dgt( ctx, b, INVERSE );

	CUDAEngine::execute_approx_modulus_reduction(
		ctx,
		a->d_coefs,
		b->d_coefs,
		level);

	a->base = QBase;
	a->state = RNSSTATE;

}

__host__ void poly_rho_ckks(
	Context *ctx, 
	poly_t *b,
	poly_t *a){
	poly_init(ctx, a);
	poly_init(ctx, b);

	////////////////////
	// Computes xi() //
	////////////////////

	CUDAEngine::execute_rho_ckks_rns( b->d_coefs, a->d_coefs, ctx );

	b->state = a->state;
}

__host__ void poly_ckks_rescale(
	Context *ctx, 
	poly_t *a,
	poly_t *b,
	int level){
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(a->base == QBase);

	////////////////////
	// Computes xi() //
	////////////////////

	DGTEngine::execute_dgt(ctx, a, INVERSE);
	DGTEngine::execute_dgt(ctx, b, INVERSE);

	CUDAEngine::execute_ckks_rescale(
		a->d_coefs,
		b->d_coefs,
		level,
		ctx
		);
}

__host__  void poly_add(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	poly_init(ctx, a);
	poly_init(ctx, b);
	poly_init(ctx, c);
	
	if(a->state != b->state){
		DGTEngine::execute_dgt(ctx, a, INVERSE);
		DGTEngine::execute_dgt(ctx, b, INVERSE);
	}

	DGTEngine::execute_add_dgt(
		c->d_coefs,
		a->d_coefs,
		b->d_coefs,
		a->base,
		ctx->get_stream()
		);

	c->state = a->state;
	c->base = a->base;
}

__host__  void poly_double_add(
	Context *ctx,
	poly_t *c1, poly_t *a1, poly_t *b1,
	poly_t *c2, poly_t *a2, poly_t *b2){
	poly_init(ctx, c1);
	poly_init(ctx, a1);
	poly_init(ctx, b1);
	poly_init(ctx, c2);
	poly_init(ctx, a2);
	poly_init(ctx, b2);
	
	if(a1->state != b1->state){
		DGTEngine::execute_dgt(ctx, a1, INVERSE);
		DGTEngine::execute_dgt(ctx, b1, INVERSE);
	}
	if(a2->state != b2->state){
		DGTEngine::execute_dgt(ctx, a2, INVERSE);
		DGTEngine::execute_dgt(ctx, b2, INVERSE);
	}

	DGTEngine::execute_double_add_dgt(
		c1->d_coefs, a1->d_coefs, b1->d_coefs,
		c2->d_coefs, a2->d_coefs, b2->d_coefs, 
		a1->base,
		ctx->get_stream()
		);

	c1->state = a1->state;
	c2->state = a2->state;
	c1->base = a1->base;
	c2->base = a2->base;
}

__host__  void poly_sub(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	poly_init(ctx, a);
	poly_init(ctx, b);
	poly_init(ctx, c);
	
	if(a->state != b->state){
		DGTEngine::execute_dgt(ctx, a, INVERSE);
		DGTEngine::execute_dgt(ctx, b, INVERSE);
	}

	DGTEngine::execute_sub_dgt(
		c->d_coefs,
		a->d_coefs,
		b->d_coefs,
		a->base,
		ctx->get_stream()
		);

	c->state = a->state;
	c->base = a->base;
}

__host__  void poly_mul(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	poly_init(ctx, a);
	poly_init(ctx, b);
	poly_init(ctx, c);

	DGTEngine::execute_dgt(ctx, a, FORWARD);
	DGTEngine::execute_dgt(ctx, b, FORWARD);

	DGTEngine::execute_mul_dgt_gi(
		c->d_coefs,
		a->d_coefs,
		b->d_coefs,
		a->base,
		ctx->get_stream()
		);

	c->state = TRANSSTATE;
	c->base = a->base;
}

__host__  void poly_mul_add(
	Context *ctx,
	poly_t *d, poly_t *a, poly_t *b, poly_t *c){
	
	poly_init(ctx, d);
	poly_init(ctx, a);
	poly_init(ctx, b);
	poly_init(ctx, c);

	DGTEngine::execute_dgt(ctx, a, FORWARD);
	DGTEngine::execute_dgt(ctx, b, FORWARD);
	DGTEngine::execute_dgt(ctx, c, FORWARD);
	d->base = a->base;
	d->state = TRANSSTATE;

	DGTEngine::execute_muladd_dgt(
		d->d_coefs,
		a->d_coefs, b->d_coefs, c->d_coefs,
		d->base,
		ctx->get_stream()
		);


}


__host__  void poly_dr2(
	Context *ctx,
	poly_t *ct21, // Outcome
	poly_t *ct22, // Outcome
	poly_t *ct23, // Outcome
	poly_t *ct01, // Operand 1
	poly_t *ct02, // Operand 1
	poly_t *ct11, // Operand 2
	poly_t *ct12){// Operand 2

	poly_init(ctx, ct21);
	poly_init(ctx, ct22);
	poly_init(ctx, ct23);
	poly_init(ctx, ct01);
	poly_init(ctx, ct02);
	poly_init(ctx, ct11);
	poly_init(ctx, ct12);

	DGTEngine::execute_dgt(ctx, ct01, FORWARD);
	DGTEngine::execute_dgt(ctx, ct02, FORWARD);
	DGTEngine::execute_dgt(ctx, ct11, FORWARD);
	DGTEngine::execute_dgt(ctx, ct12, FORWARD);

	DGTEngine::execute_dr2_dgt(
		ct21->d_coefs,
		ct22->d_coefs,
		ct23->d_coefs,
		ct01->d_coefs,
		ct02->d_coefs,
		ct11->d_coefs,
		ct12->d_coefs,
		ct01->base,
		ctx->get_stream()
		);

	ct21->state = TRANSSTATE;
	ct22->state = TRANSSTATE;
	ct23->state = TRANSSTATE;

	ct21->base = ct01->base;
	ct22->base = ct01->base;
	ct23->base = ct01->base;

}

__host__  void poly_mul_int(
	Context *ctx, 
	poly_t *c,
	poly_t *a,
	uint64_t b){
	poly_init(ctx, a);
	poly_init(ctx, c);

	DGTEngine::execute_dgt(ctx, a, INVERSE);
	CUDAEngine::execute_polynomial_op_by_int(
		c->d_coefs, a->d_coefs, b, a->base,
		MUL, ctx);

	c->state = a->state;
	c->base = a->base;
}

__host__  void poly_double_mul(
	Context *ctx, 
	poly_t *c,	poly_t *a,	uint64_t b,
	poly_t *f,	poly_t *d,	uint64_t e){
	poly_init(ctx, a);
	poly_init(ctx, d);
	poly_init(ctx, c);
	poly_init(ctx, f);

	DGTEngine::execute_dgt(ctx, a, INVERSE);
	DGTEngine::execute_dgt(ctx, d, INVERSE);
	CUDAEngine::execute_polynomial_double_op_by_int(
		c->d_coefs, a->d_coefs, 
		f->d_coefs, d->d_coefs, 
		b, e,
		MULMUL, a->base, ctx);

	c->state = a->state;
	c->base = a->base;
}

__host__  void poly_add_int(
	Context *ctx, 
	poly_t *c,
	poly_t *a,
	uint64_t b){
	poly_init(ctx, a);
	poly_init(ctx, c);

	DGTEngine::execute_dgt(ctx, a, FORWARD);
	CUDAEngine::execute_polynomial_op_by_int(
		c->d_coefs, a->d_coefs, b, a->base, ADD, ctx);

	c->state = a->state;
	c->base = a->base;
}

__host__  void poly_sub_int(
	Context *ctx, 
	poly_t *c,
	poly_t *a,
	uint64_t b){
	poly_init(ctx, a);
	poly_init(ctx, c);

	DGTEngine::execute_dgt(ctx, a, FORWARD);
	CUDAEngine::execute_polynomial_op_by_int(
		c->d_coefs, a->d_coefs, b, a->base, SUB, ctx);

	c->state = a->state;
	c->base = a->base;
}

__host__ void poly_copy(Context *ctx, poly_t *b, poly_t *a){
	poly_init(ctx, a);
	poly_init(ctx, b);

	cudaMemcpyAsync(
		b->d_coefs,
		a->d_coefs,
		poly_get_size(a->base),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
	);
	cudaCheckError();

	b->state = a->state;
	b->base = a->base;
}

__host__ void poly_negate(Context *ctx, poly_t *a){
		
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	CUDAEngine::execute_polynomial_negate(
		a->d_coefs,
		a->d_coefs,
		a->base,
		ctx);

}

__host__ bool poly_are_equal(Context *ctx, poly_t *a, poly_t *b){
	poly_init(ctx, a);
	poly_init(ctx, b);

	if(a->base != b->base)
		return false;

	uint64_t *h_a = poly_copy_to_host(ctx, a);
	uint64_t *h_b = poly_copy_to_host(ctx, b);

	for(int i = 0; i < 2 * CUDAEngine::N * CUDAEngine::get_n_residues(a->base); i++)
		if(h_a[i] != h_b[i])
			return false;

	free(h_a);
	free(h_b);
	return true;
}

std::string poly_residue_to_string(Context *ctx, poly_t *a, int id){
	poly_init(ctx, a);

	uint64_t *h_coefs = poly_copy_to_host(ctx, a);

	// Indexes
	// If idx < 0, print all residues
	int lidx = (id >=0 ? id * 2 * CUDAEngine::N : 0);
	int hidx = (id >=0 ? (id + 1) * 2 * CUDAEngine::N : CUDAEngine::get_n_residues(QBBase) * 2 * CUDAEngine::N);

	std::ostringstream oss;
	for(int i = lidx; i < hidx; i++)
		oss << h_coefs[i] << ",";
	
	free(h_coefs);
	return oss.str();
}

__host__ std::string poly_export(Context *ctx, poly_t *p){
	return poly_residue_to_string(ctx, p);
}

__host__ poly_t* poly_import(Context *ctx, std::string s){
	// String to vector
	std::vector<NTL::ZZ> v;
	if(s.length() > 0){
		std::string value = "";
		for(int i = s.length(); i >= -1; i--)
			if(i == -1 || s[i] == ','){
				v.insert(v.begin(), NTL::to_ZZ(value.c_str()));
				value = "";
			}else if(s[i] != ' ')
				value = s[i] + value;
	}

	// Vector to poly_t
	poly_t *p = new poly_t;
	poly_init(ctx, p);
	uint64_t *h_coefs = (uint64_t*) malloc(poly_get_size(QBase));

	for (unsigned int i = 0; i < v.size(); i++)
   		h_coefs[i] = NTL::conv<uint64_t>(v[i]);
   	poly_copy_to_device(ctx, p, h_coefs);
   	cudaStreamSynchronize(ctx->get_stream());
   	free(h_coefs);
   	return p;
}

__host__ void poly_dot(
	Context *ctx,
	poly_t *c,
	poly_t *a,
	poly_t *b,
	const int k){
	// Dot product

	poly_mul(ctx, c, &a[0], &b[0]);
	for(int i = 1; i < k; i++)
		poly_mul_add(ctx, c, &a[i], &b[i], c);

}
