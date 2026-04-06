#include <newckks/arithmetic/poly_t.h>
#include <newckks/cuda/manager.h>
#include <newckks/cuda/htrans/ntt.h>
#include <newckks/ckks/ckkscontext.h>

size_t poly_get_size(poly_bases b){
	return CUDAManager::N * CUDAManager::get_n_residues(b) * sizeof(uint64_t);
}

 void poly_init(Context *ctx, poly_t *a, poly_bases b){
	if(a->state != NONINITIALIZED)
		return;
	assert(ctx);
	assert(CUDAManager::is_init);

	// Device
	if(hasSupportStreamAlloc()){
		cudaMallocAsync((void**)&a->d_coefs, poly_get_size(b), ctx->get_stream());
		cudaCheckError();
		cudaMemsetAsync(a->d_coefs, 0, poly_get_size(b), ctx->get_stream());
		cudaCheckError();
	} else{
		cudaMalloc((void**)&a->d_coefs, poly_get_size(b));
		cudaCheckError();
		cudaMemset(a->d_coefs, 0, poly_get_size(b));
		cudaCheckError();
	}

	a->state = RNSSTATE;
	a->base = b;
}

 void poly_free(Context *ctx, poly_t *a){
	if(!a || a->state == NONINITIALIZED)
		return;

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

 void poly_clear(Context *ctx, poly_t *a){
	assert(CUDAManager::is_init);
	poly_init(ctx, a);

	cudaMemsetAsync(a->d_coefs,0, poly_get_size(a->base), ctx->get_stream());
	cudaCheckError();

}

void poly_copy_to_device(Context *ctx, poly_t *a, uint64_t *h_coefs){
	poly_init(ctx, a);
	assert(h_coefs);

	cudaMemcpyAsync(
		a->d_coefs,
		h_coefs,
		poly_get_size(a->base),
		cudaMemcpyHostToDevice,
		ctx->get_stream()
	);
	cudaCheckError();

	a->state = RNSSTATE;
}

uint64_t* poly_copy_to_host(Context *ctx, poly_t *a){
	poly_init(ctx, a);

	// Temp
	poly_t b;
	poly_copy(ctx, &b, a);

	COMMONEngine::execute( ctx, &b, INVERSE);

	// The output
	uint64_t *h_coefs = (uint64_t*) malloc(poly_get_size(a->base));
	cudaMemcpyAsync(
		h_coefs,
		b.d_coefs,
		poly_get_size(a->base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);
	cudaCheckError();

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	poly_free(ctx, &b);

	return h_coefs;
}

void poly_copy(Context *ctx, poly_t *b, poly_t *a){
	assert(a->state != NONINITIALIZED);
	poly_init(ctx, b, a->base);

	cudaMemcpyAsync(
		b->d_coefs,
		a->d_coefs,
		poly_get_size(a->base),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
	);
	cudaCheckError();

	b->state = a->state;
}

bool poly_are_equal(Context *ctx, poly_t *a, poly_t *b){
	poly_init(ctx, a);
	poly_init(ctx, b);

	if(a->base != b->base)
		return false;

	uint64_t *h_a = poly_copy_to_host(ctx, a);
	uint64_t *h_b = poly_copy_to_host(ctx, b);
	
	for(int i = 0; i < CUDAManager::N * CUDAManager::get_n_residues(a->base); i++)
		if(h_a[i] != h_b[i])
			return false;
		
	free(h_a);
	free(h_b);
	return true;
}

void poly_add(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	poly_bases base = std::min(a->base, b->base);
	poly_init(ctx, a, base);
	poly_init(ctx, b, base);
	poly_init(ctx, c, base);

	if(a->state != b->state){
		COMMONEngine::execute(ctx, a, INVERSE);
		COMMONEngine::execute(ctx, b, INVERSE);
	}

	COMMONEngine::execute_op(
		ctx,
		c->d_coefs, a->d_coefs, b->d_coefs,
		ADDOP, base);

	c->state = a->state;
	c->base = base;
}

void poly_add(Context *ctx, poly_t *c, poly_t *a, uint64_t b){
	poly_init(ctx, a);
	poly_init(ctx, c, a->base);

	COMMONEngine::execute(ctx, a, FORWARD);
	COMMONEngine::execute_op_by_uint(
		ctx, c->d_coefs, a->d_coefs, b, ADDOP, a->base);

	c->state = a->state;
	c->base = a->base;
}

void poly_sub(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	poly_bases base = std::min(a->base, b->base);
	poly_init(ctx, a, base);
	poly_init(ctx, b, base);
	poly_init(ctx, c, base);

	if(a->state != b->state){
		COMMONEngine::execute(ctx, a, INVERSE);
		COMMONEngine::execute(ctx, b, INVERSE);
	}

	COMMONEngine::execute_op(
		ctx,
		c->d_coefs, a->d_coefs, b->d_coefs,
		SUBOP, base);

	c->state = a->state;
	c->base = base;
}

void poly_sub(Context *ctx, poly_t *c, poly_t *a, uint64_t b){
	poly_init(ctx, a);
	poly_init(ctx, c);

	COMMONEngine::execute(ctx, a, FORWARD);
	COMMONEngine::execute_op_by_uint(
		ctx, c->d_coefs, a->d_coefs, b, SUBOP, a->base);

	c->state = a->state;
	c->base = a->base;
}

void poly_mul(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	poly_bases base = std::min(a->base, b->base);
	poly_init(ctx, a, base);
	poly_init(ctx, b, base);
	poly_init(ctx, c, base);

	COMMONEngine::execute(ctx, a, FORWARD);
	COMMONEngine::execute(ctx, b, FORWARD);

	COMMONEngine::execute_op(
		ctx,
		c->d_coefs,
		a->d_coefs,
		b->d_coefs,
		MULOP,
		base);

	c->state = a->state;
	c->base = base;
}

void poly_mul(Context *ctx, poly_t *c, poly_t *a, uint64_t b){
	poly_init(ctx, a);
	poly_init(ctx, c);

	COMMONEngine::execute(ctx, a, INVERSE);
	COMMONEngine::execute_op_by_uint(
		ctx,
		c->d_coefs, a->d_coefs, b,
		MULOP, 
		a->base);

	c->state = a->state;
	c->base = a->base;
}

void poly_add_add(
	Context *ctx,
	poly_t *c1, poly_t *a1, poly_t *b1,
	poly_t *c2, poly_t *a2, poly_t *b2){

	poly_bases base = std::min(
		std::min(a1->base, b1->base), 
		std::min(a2->base, b2->base));
	poly_init(ctx, a1, base);
	poly_init(ctx, b1, base);
	poly_init(ctx, a2, base);
	poly_init(ctx, b2, base);
	poly_init(ctx, c1, base);
	poly_init(ctx, c2, base);
	
	if(a1->state != b1->state){
		COMMONEngine::execute(ctx, a1, INVERSE);
		COMMONEngine::execute(ctx, b1, INVERSE);
	}
	if(a2->state != b2->state){
		COMMONEngine::execute(ctx, a2, INVERSE);
		COMMONEngine::execute(ctx, b2, INVERSE);
	}

	COMMONEngine::execute_dualop(
		ctx,
		c1->d_coefs, a1->d_coefs, b1->d_coefs,
		c2->d_coefs, a2->d_coefs, b2->d_coefs, 
		ADDADDOP, base);

	c1->state = a1->state;
	c1->base = base;
	c2->state = a2->state;
	c2->base = base;

}

 void poly_mul_add(
	Context *ctx, poly_t *d,
	poly_t *a, poly_t *b, poly_t *c){
	
	poly_bases base = std::min(std::min(a->base, b->base), c->base);
	poly_init(ctx, d, base);
	poly_init(ctx, a, base);
	poly_init(ctx, b, base);
	poly_init(ctx, c, base);

	COMMONEngine::execute(ctx, a, FORWARD);
	COMMONEngine::execute(ctx, b, FORWARD);
	COMMONEngine::execute(ctx, c, FORWARD);
	
	COMMONEngine::execute_seqop(
		ctx,
		d->d_coefs,
		a->d_coefs,
		b->d_coefs,
		c->d_coefs,
		MULandADDOP,
		base);

	d->state = a->state;
	d->base = base;

}

 void poly_dr2(
    Context *ctx,
    poly_t *ct21, // Outcome
    poly_t *ct22, // Outcome
    poly_t *ct23, // Outcome
    poly_t *ct01, // Operand 1
    poly_t *ct02, // Operand 1
    poly_t *ct11, // Operand 2
    poly_t *ct12){// Operand 2

	poly_bases base = ct01->base;
	poly_init(ctx, ct21, base);
	poly_init(ctx, ct22, base);
	poly_init(ctx, ct23, base);
	poly_init(ctx, ct01, base);
	poly_init(ctx, ct02, base);
	poly_init(ctx, ct11, base);
	poly_init(ctx, ct12, base);

	COMMONEngine::execute(ctx, ct01, FORWARD);
	COMMONEngine::execute(ctx, ct02, FORWARD);
	COMMONEngine::execute(ctx, ct11, FORWARD);
	COMMONEngine::execute(ctx, ct12, FORWARD);

	CUDAManager::execute_dr2(
		ctx,
		ct21->d_coefs,
		ct22->d_coefs,
		ct23->d_coefs,
		ct01->d_coefs,
		ct02->d_coefs,
		ct11->d_coefs,
		ct12->d_coefs);

	ct21->state = TRANSSTATE;
	ct21->base = base;
	ct22->state = TRANSSTATE;
	ct22->base = base;
	ct23->state = TRANSSTATE;
	ct23->base = base;

}

void poly_modup( Context *ctx, poly_t *a, poly_t *b, int level){
	poly_init(ctx, a, QBBase);
	poly_init(ctx, b);

	assert(a->base == QBBase);
	assert(b->base == QBase);

	poly_copy(ctx, a, b);
	COMMONEngine::execute(ctx, a, INVERSE);

	CUDAManager::execute_modup(
		ctx,
		a->d_coefs,
		a->d_coefs,
		level);

}

void poly_moddown( Context *ctx, poly_t *a, poly_t *b, int level){
	poly_init(ctx, a, QBase);
	poly_init(ctx, b, QBBase);

	assert(b->base == QBBase);

	COMMONEngine::execute(ctx, b, INVERSE);

	CUDAManager::execute_moddown(
		ctx,
		a->d_coefs,
		b->d_coefs,
		level);

	a->state = RNSSTATE;
}

__host__ void poly_rho(
	Context *ctx, 
	poly_t *b,
	poly_t *a){
	poly_init(ctx, a);
	poly_init(ctx, b);

	////////////////////
	// Computes xi() //
	////////////////////

	CUDAManager::execute_rho(ctx, b->d_coefs, a->d_coefs);
	b->state = a->state;
}

void poly_rescale( Context *ctx, poly_t *a, poly_t *b, int level){
	poly_init(ctx, a);
	poly_init(ctx, b);

	COMMONEngine::execute(ctx, a, INVERSE);
	COMMONEngine::execute(ctx, b, INVERSE);

	CUDAManager::execute_rescale(
		ctx,
		a->d_coefs,
		b->d_coefs,
		level);

	a->state = RNSSTATE;
	b->state = RNSSTATE;

}

void poly_negate( Context *ctx, poly_t *a){
	poly_init(ctx, a);

	COMMONEngine::execute_op(
		ctx,
		a->d_coefs,
		a->d_coefs,
		NULL,
		NEGATEOP,
		a->base);

}

std::string poly_residue_to_string(Context *ctx, poly_t *a, int id){
	poly_init(ctx, a);

	COMMONEngine::execute(ctx, a, INVERSE);
	uint64_t *h_coefs = poly_copy_to_host(ctx, a);

	// Indexes
	// If idx < 0, print all residues
	int lidx = (id >=0 ? id * CUDAManager::N : 0);
	int hidx = (id >=0 ? (id + 1) * CUDAManager::N : CUDAManager::get_n_residues(a->base) * CUDAManager::N);

	std::ostringstream oss;
	for(int i = lidx; i < hidx; i++)
		oss << h_coefs[i] << ",";
	
	free(h_coefs);
	return oss.str();
}

std::string poly_export(Context *ctx, poly_t *p){
	return poly_residue_to_string(ctx, p);
}

poly_t* poly_import(Context *ctx, std::string s){
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
	uint64_t *h_coefs = (uint64_t*) malloc(poly_get_size(p->base));

	for (unsigned int i = 0; i < v.size(); i++)
   		h_coefs[i] = NTL::conv<uint64_t>(v[i]);
   	poly_copy_to_device(ctx, p, h_coefs);
   	cudaStreamSynchronize(ctx->get_stream());
   	free(h_coefs);
   	return p;
}