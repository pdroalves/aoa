#include <AOADGT/ckks.h>
#include <openssl/sha.h>


void print_hash(CKKSContext *ctx, poly_t *p, int i){
	std:string residue = poly_residue_to_string(ctx, p, i);
	unsigned char obuf[20];

	SHA1((unsigned char*)residue.c_str(), strlen(residue.c_str()), obuf);
	for (i = 0; i < 20; i++) 
    	printf("%02x ", obuf[i]);
    std::cout << std::endl;
}

void print_residue(CKKSContext *ctx, poly_t *p, int i){
    std::cout << poly_residue_to_string(ctx, p, i) << std::endl;
}

__host__ Keys* ckks_keygen(CKKSContext *ctx, SecretKey *sk){
	Keys *keys;

	//////////////////////////////////
	// Alloc memory for each key //
	//////////////////////////////////
	keys = new Keys;

	////////////////
	// Public key //
	////////////////
	keys->pk = ckks_new_pk(ctx, sk);

	//////////////////////
	// Computation keys //
	//////////////////////
	sk->s.base = QBBase; // This is a hack
	poly_t s2;
	poly_init(ctx, &s2, QBBase);

	// Evaluation key //
	poly_mul(ctx, &s2, &sk->s, &sk->s);
	keys->evk = ckks_new_evk(ctx, &s2, &sk->s);
	poly_clear(ctx, &s2);

	// Rotation key //
	for (int i = 1; i < CUDAEngine::N; i *= 2){
		rotate_slots_right(ctx, &s2, &sk->s, i);
		keys->rtk_right.insert(
			pair<int, RotationKey*>(i, ckks_new_evk(ctx, &s2, &sk->s))
			);
	}
	for (int i = 1; i < CUDAEngine::N; i *= 2){
		rotate_slots_left(ctx, &s2, &sk->s, i);
		keys->rtk_left.insert(
			pair<int, RotationKey*>(i, ckks_new_evk(ctx, &s2, &sk->s))
			);
	}

	// Conjugation key //
	poly_copy(ctx, &s2, &sk->s);
	conjugate_slots(ctx, &s2);
	keys->cjk = ckks_new_evk(ctx, &s2, &sk->s);

	poly_free(ctx, &s2);
	sk->s.base = QBase; // This is a hack

	////////////
	// Export //
	////////////
	ctx->pk = keys->pk;
	ctx->evk = keys->evk;
	ctx->rtk_right = keys->rtk_right;
	ctx->rtk_left = keys->rtk_left;
	ctx->cjk = keys->cjk;

	return keys;
}

__host__ cipher_t* ckks_encrypt_poly(CKKSContext *ctx, poly_t *m){
    	/////////////////////
	// Allocate memory //
	/////////////////////
	cipher_t *ct = new cipher_t;
	cipher_init(ctx, ct);

    return ckks_encrypt_poly(ctx, ct, m);
}

__host__ cipher_t* ckks_encrypt_poly(CKKSContext *ctx, cipher_t *ct, poly_t *m){
	assert(CUDAEngine::is_init);
	cipher_init(ctx, ct);

	Sampler::sample(ctx, ctx->u, ZO);
	Sampler::sample(ctx, ctx->e1, DISCRETE_GAUSSIAN);
	Sampler::sample(ctx, ctx->e2, DISCRETE_GAUSSIAN);

	
	// ct = v * pk + (m + e1, e2)
	poly_add(ctx, ctx->e1, ctx->e1, m);
	poly_mul_add(ctx, &ct->c[0], ctx->u, &ctx->pk->b, ctx->e1);
	poly_mul_add(ctx, &ct->c[1], ctx->u, &ctx->pk->a, ctx->e2);

	poly_clear(ctx, ctx->u);
	poly_clear(ctx, ctx->e1);
	poly_clear(ctx, ctx->e2);
	return ct;

}

__host__ cipher_t* ckks_encrypt(
	CKKSContext *ctx,
	complex<double>* val,
	int slots,
	int empty_slots){

	/////////////////////
	// Allocate memory //
	/////////////////////
	cipher_t *ct = new cipher_t;
	cipher_init(ctx, ct);

	// Encode
	ctx->encode(ctx->m, &ct->scale, val, slots, empty_slots);
	assert(ctx->m->state != NONINITIALIZED);

	// Encrypt
	ckks_encrypt_poly(ctx, ct, ctx->m);
	ct->slots = slots + empty_slots;
	poly_clear(ctx, ctx->m);

	return ct;
}

__host__ cipher_t* ckks_encrypt(
	CKKSContext *ctx,
	cipher_t *ct,
	complex<double>* val,
	int slots,
	int empty_slots){

	// Encode
	ctx->encode(ctx->m, &ct->scale, val, slots, empty_slots);
	assert(ctx->m->state != NONINITIALIZED);

	// Encrypt
	ckks_encrypt_poly(ctx, ct, ctx->m);
	ct->slots = slots + empty_slots;
	poly_clear(ctx, ctx->m);

	return ct;
}

__host__ cipher_t* ckks_encrypt(
	CKKSContext *ctx,
	double val,
	int slots,
	int empty_slots){

	complex<double> *cval = new complex<double>[slots];
	for (int i = 0; i < slots; i++)
		cval[i] = {val, 0};
	return ckks_encrypt(ctx, cval, slots, empty_slots);
}

__host__ cipher_t* ckks_encrypt(
	CKKSContext *ctx,
	cipher_t *ct,
	double val,
	int slots,
	int empty_slots){

	complex<double> *cval = new complex<double>[slots];
	for (int i = 0; i < slots; i++)
		cval[i] = {val, 0};
	return ckks_encrypt(ctx, ct, cval, slots, empty_slots);
}

__host__ poly_t* ckks_decrypt_poly(
	CKKSContext *ctx, 
	poly_t *m, 
	cipher_t *c, 
	SecretKey *sk){
	assert(CUDAEngine::is_init);

	//////////////////////////////////////////
	// Compute x = |(c0 + c1*s)|_q //
	//////////////////////////////////////////
	poly_mul_add(ctx, m, &c->c[1], &sk->s, &c->c[0]);

	return m;
}


__host__ complex<double>* ckks_decrypt(
	CKKSContext *ctx,
	complex<double> *val,
	cipher_t *c,
	SecretKey *sk){
	assert(CUDAEngine::is_init);

	// poly_clear(ctx, ctx->m);
	ckks_decrypt_poly(ctx, ctx->m, c, sk);
	// std::cout << "Will decrypt: " << poly_residue_to_string(ctx, ctx->m, 0) << std::endl;
	// Decode
	if(c->slots == 1)
		ctx->decodeSingle(val, ctx->m, c->scale);
	else
		ctx->decode(val, ctx->m, c->scale, c->slots);		
	return val;
}

__host__ complex<double>* ckks_decrypt(
	CKKSContext *ctx,
	cipher_t *c,
	SecretKey *sk){
	assert(CUDAEngine::is_init);

	complex<double> *val = new complex<double>[c->slots];

	return ckks_decrypt(ctx, val, c, sk);
}

__host__ void ckks_rescale(
	CKKSContext *ctx, 
	cipher_t *c){

	if(c-> level == 0)
		throw std::runtime_error(
			"The ciphertext level is 0. No more rescaling is possible."
			);
	poly_ckks_rescale(ctx, &c->c[0], &c->c[1], c->level);
	c->level--;
}

cipher_t* ckks_add(CKKSContext *ctx, cipher_t *c1, cipher_t *c2){
	cipher_t *c3 = new cipher_t;

	ckks_add(ctx, c3, c1, c2);

	return c3;
}

__host__  void ckks_add(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2){
	assert(CUDAEngine::is_init);
	assert(c1->slots == c2->slots);

	poly_double_add(ctx, &c3->c[0], &c1->c[0], &c2->c[0], &c3->c[1], &c1->c[1], &c2->c[1]);

	c3->level = min(c1->level, c2->level);
	c3->slots = c1->slots;
	c3->scale = c1->scale;
	return;
}

cipher_t* ckks_sub(CKKSContext *ctx, cipher_t *c1, cipher_t *c2){
	cipher_t *c3 = new cipher_t;

	ckks_sub(ctx, c3, c1, c2);

	return c3;
}

__host__  void ckks_sub(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2){
	assert(CUDAEngine::is_init);
	assert(c1->slots == c2->slots);

	poly_sub(ctx, &c3->c[0], &c1->c[0], &c2->c[0]);
	poly_sub(ctx, &c3->c[1], &c1->c[1], &c2->c[1]);

	c3->level = min(c1->level, c2->level);
	c3->slots = c1->slots;
	c3->scale = c1->scale;
	return;
}

__host__ void ckks_mul_without_rescale(
	CKKSContext *ctx,
	cipher_t *c3,
	cipher_t *c1,
	cipher_t *c2){
	assert(CUDAEngine::is_init);
	assert(c1->c[0].state != NONINITIALIZED);
	assert(c2->c[0].state != NONINITIALIZED);

	if(c1->level == 0 || c2->level == 0)
		throw std::runtime_error(
			"The ciphertext level is 0. No more rescaling is possible."
			);

	if(c1->slots != c2->slots)
		throw std::runtime_error(
			"Ciphertexts must have the same number of slots."
			);
	c3->level = min(c1->level, c2->level);
	////////////////////////////////////////////
	// Execute the homomorphic multiplication //
	////////////////////////////////////////////
	
	////////////////////////
	// Compute DR2(c1,c2) //
	////////////////////////
	//poly_mul(ctx, &ctx->axbx2, &c1->c[0], &c2->c[1]);
	//poly_mul_add(ctx, &ctx->axbx1, &c1->c[1], &c2->c[0], &ctx->axbx2); // d1

	poly_double_add(
		ctx,
		&ctx->axbx1, &c1->c[1], &c1->c[0],
		&ctx->axbx2, &c2->c[1], &c2->c[0]
		);
	poly_mul(ctx, &ctx->axbx1, &ctx->axbx1, &ctx->axbx2); // d1
	poly_mul(ctx, &ctx->bxbx, &c1->c[0], &c2->c[0]); // d0
	poly_mul(ctx, &ctx->axax, &c1->c[1], &c2->c[1]); // d2


	// ModUp
	poly_modup(ctx, ctx->d2axaxQB, &ctx->axax, c3->level);

	// Multiply \tilde{d2} by the evk
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->evk->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->evk->a);

	// // ModDown
	poly_moddown(ctx, &ctx->d_tildeQ[0], &ctx->d_tildeQB[0], c3->level);
	poly_moddown(ctx, &ctx->d_tildeQ[1], &ctx->d_tildeQB[1], c3->level);

	// Output
	poly_double_add(ctx,
		&c3->c[0], &ctx->bxbx, &ctx->d_tildeQ[0],
		&c3->c[1], &ctx->axbx1, &ctx->d_tildeQ[1]);
	poly_sub(ctx, &c3->c[1], &c3->c[1], &ctx->bxbx);
	poly_sub(ctx, &c3->c[1], &c3->c[1], &ctx->axax);


	c3->slots = c1->slots; 
	c3->scale = conv<uint64_t>(
			to_ZZ(c1->scale) * to_ZZ(c2->scale) / CUDAEngine::RNSPrimes[c3->level]
			// (to_ZZ(c1->scale) + to_ZZ(c2->scale)) / 2
			// c1->scale
			// sqrt(to_RR(c1->scale) * to_RR(c2->scale))

		);
	// std::cout << "C1 scale: " << to_ZZ(c1->scale) << ", C2 scale: " << to_ZZ(c2->scale) << ", C3 scale: " << to_ZZ(c3->scale) << std::endl;
	return;
}



__host__ void ckks_mul(
	CKKSContext *ctx,
	cipher_t *c3,
	cipher_t *c1,
	cipher_t *c2){
	assert(CUDAEngine::is_init);

	////////////////////////////////////////////
	// Execute the homomorphic multiplication //
	////////////////////////////////////////////
	ckks_mul_without_rescale(ctx, c3, c1, c2);
	ckks_rescale(ctx, c3);
	return;
}

__host__ cipher_t* ckks_mul(CKKSContext *ctx, cipher_t *c1, cipher_t *c2){
	assert(CUDAEngine::is_init);

	cipher_t *c3 = new cipher_t;
	cipher_init(ctx, c3);

	ckks_mul(ctx, c3, c1, c2);

	return c3;
}

__host__ void ckks_mul(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	poly_t *val){

	ckks_mul_without_rescale(ctx, c2, c1, val);
	ckks_rescale(ctx, c2);
	return;
}

__host__ cipher_t* ckks_mul(
	CKKSContext *ctx,
	cipher_t *c1,
	poly_t *val){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_mul(ctx, c2, c1, val);

	return c2;
}

__host__ void ckks_mul_without_rescale(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	poly_t *val){

	assert(CUDAEngine::is_init);

	poly_mul(ctx, &c2->c[0], &c1->c[0], val);
	poly_mul(ctx, &c2->c[1], &c1->c[1], val);

	c2->level = c1->level;
	c2->slots = c1->slots;
	c2->scale = NTL::conv<uint64_t>(
		NTL::to_ZZ(c1->scale) * NTL::to_ZZ(c1->scale) / CUDAEngine::RNSPrimes[c1->level]
		);

	return;
}

__host__ cipher_t* ckks_square(CKKSContext *ctx, cipher_t *c1){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_square(ctx, c2, c1);

	return c2;
}

__host__ void ckks_square_without_rescale(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1){
	assert(CUDAEngine::is_init);

	////////////////////////////////////////////
	// Execute the homomorphic multiplication //
	////////////////////////////////////////////
	
	////////////////////////
	// Compute DR2(c1,c2) //
	////////////////////////
	poly_add(ctx, &ctx->axbx1, &c1->c[1], &c1->c[0]);
	poly_mul(ctx, &ctx->axbx1, &ctx->axbx1, &ctx->axbx1);
	poly_mul(ctx, &ctx->bxbx, &c1->c[0], &c1->c[0]);
	poly_mul(ctx, &ctx->axax, &c1->c[1], &c1->c[1]);

	// ModUp
	poly_modup(ctx, ctx->d2axaxQB, &ctx->axax, c1->level);

	// Multiply \tilde{d2} by the evk
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->evk->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->evk->a);

	// // ModDown
	poly_moddown(ctx, &ctx->d_tildeQ[0], &ctx->d_tildeQB[0], c1->level);
	poly_moddown(ctx, &ctx->d_tildeQ[1], &ctx->d_tildeQB[1], c1->level);

	// Output
	poly_double_add(ctx,
		&c2->c[1], &ctx->axbx1, &ctx->d_tildeQ[1],
		&c2->c[0], &ctx->bxbx, &ctx->d_tildeQ[0]);
	poly_sub(ctx, &c2->c[1], &c2->c[1], &ctx->bxbx);
	poly_sub(ctx, &c2->c[1], &c2->c[1], &ctx->axax);

	c2->level = c1->level;
	c2->slots = c1->slots; 
	c2->scale = conv<uint64_t>(
			to_ZZ(c1->scale) * to_ZZ(c1->scale) / CUDAEngine::RNSPrimes[c2->level]
		);
	return;
}

__host__ void ckks_square(CKKSContext *ctx, cipher_t *c2, cipher_t *c1){
	assert(CUDAEngine::is_init);

	//////////////////////////////////////
	// Execute the homomorphic squaring //
	//////////////////////////////////////
	ckks_square_without_rescale(ctx, c2, c1);
	ckks_rescale(ctx, c2);
	return;
}

__host__ void ckks_inner_prod(
	CKKSContext *ctx,
	cipher_t *c3,
	cipher_t *c1,
	cipher_t *c2,
	const int size){
	assert(CUDAEngine::is_init);
	assert(c1->slots == c2->slots);

	cipher_init(ctx, c3);
	c3->slots = c1->slots;
	for(int i = 0; i < size; i++){
		ckks_mul_without_rescale(ctx, &ctx->aux[0], &c1[i], &c2[i]);
		ckks_add(ctx, c3, c3, &ctx->aux[0]);
	}

	ckks_rescale(ctx, c3);
	return;
}

__host__ void ckks_inner_prod(
	CKKSContext *ctx,
	cipher_t *c3,
	cipher_t *c1,
	double *c2,
	const int size){
	assert(CUDAEngine::is_init);

	cipher_init(ctx, c3);
	c3->slots = c1->slots;
	for(int i = 0; i < size; i++){
		ckks_mul_without_rescale(ctx, &ctx->aux[0], &c1[i], c2[i]);
		ckks_add(ctx, c3, c3, &ctx->aux[0]);
	}

	ckks_rescale(ctx, c3);
	return;
}


template<typename T>
__host__ void ckks_batch_inner_prod(
	CKKSContext *ctx,
	cipher_t *c3,
	cipher_t *c1,
	T *c2){
	assert(CUDAEngine::is_init);

	ckks_mul(ctx, c3, c1, c2);
	ckks_sumslots(ctx, c3, c3);
	return;
}

template __host__ void ckks_batch_inner_prod<cipher_t>(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);
template __host__ void ckks_batch_inner_prod<poly_t>(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, poly_t *c2);


__host__ void ckks_sumslots(CKKSContext *ctx, cipher_t *c2, cipher_t *c1){
	// Rotate and add to sum slots
	if(c1 != c2)
		cipher_copy(ctx, c2, c1);
	for(int i = (c2->slots >> 1); i >= 1; i /= 2){
		ckks_rotate_left(ctx, &ctx->aux[0], c2, i);
		ckks_add(ctx, c2, c2, &ctx->aux[0]);
	}
}

__host__ cipher_t* ckks_sumslots(
	CKKSContext *ctx,
	cipher_t *c){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_sumslots(ctx, c2, c);

	return c2;
}

__host__ void ckks_add(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	double val){
	
	assert(CUDAEngine::is_init);
	uint64_t cnst = abs(val) * c1->scale;

	if(val >= 0)
		poly_add_int(ctx, &c2->c[0], &c1->c[0], cnst);
	else
		poly_sub_int(ctx, &c2->c[0], &c1->c[0], cnst);

	poly_copy(ctx, &c2->c[1], &c1->c[1]);

	c2->level = c1->level;
	c2->slots = c1->slots;
	c2->scale = c1->scale;
	return;
}

__host__ cipher_t* ckks_add(
	CKKSContext *ctx,
	cipher_t *c1,
	double val){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_add(ctx, c2, c1, val);

	return c2;
}

__host__ void ckks_mul_without_rescale(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	double val){
	assert(CUDAEngine::is_init);

	uint64_t x = lrint(abs(val) * (c1->scale));

	poly_double_mul(
		ctx,
		&c2->c[0], &c1->c[0], x,
		&c2->c[1], &c1->c[1], x);

	// poly_mul_int(ctx, &c2->c[0], &c1->c[0], x);
	// poly_mul_int(ctx, &c2->c[1], &c1->c[1], x);

	if(val < 0){
		poly_negate(ctx, &c2->c[0]);
		poly_negate(ctx, &c2->c[1]);
	}
	c2->level = c1->level;
	c2->slots = c1->slots;
	c2->scale = c1->scale;

	return;
}

__host__ cipher_t* ckks_mul_without_rescale(
	CKKSContext *ctx,
	cipher_t *c1,
	double val){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_mul(ctx, c2, c1, val);

	return c2;
}

__host__ void ckks_mul(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	double val){

	ckks_mul_without_rescale(ctx, c2, c1, val);
	ckks_rescale(ctx, c2);
	return;
}

__host__ cipher_t* ckks_mul(
	CKKSContext *ctx,
	cipher_t *c1,
	double val){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_mul(ctx, c2, c1, val);

	return c2;
}

__host__ void ckks_power_of_2(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	uint64_t x){
	
	// Todo: assert x is a power of 2

	assert(CUDAEngine::is_init);
	cipher_init(ctx, c2);
	if(x == 0){
		ckks_encrypt(ctx, c2, 1, c1->slots);
		return;
	}
	cipher_clear(ctx, &ctx->aux[0]);
	cipher_copy(ctx, &ctx->aux[0], c1);
	while(x > 0){
		ckks_square(ctx, &ctx->aux[0], &ctx->aux[0]);
		x >>= 1;
	}
	return;
}

__host__ cipher_t* ckks_power_of_2(
	CKKSContext *ctx,
	cipher_t *c1,
	uint64_t x){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_power_of_2(ctx, c2, c1, x);

	return c2;
}

__host__ void ckks_power(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	uint64_t x){
	assert(CUDAEngine::is_init);
	cipher_init(ctx, c2, c1->slots);

	if(x == 0){
		ckks_encrypt(ctx, c2, 1, c1->slots);
		return;
	}

	cipher_clear(ctx, &ctx->aux[0]);
	cipher_copy(ctx, &ctx->aux[0], c1);
	bool first_op = true;
	while(x > 0){
		if(x % 2 == 1){
			if(first_op){
				ckks_add(ctx, c2, c2, &ctx->aux[0]);
				first_op = false;
			}
			else
				ckks_mul(ctx, c2, c2, &ctx->aux[0]);
		}
		x >>= 1;
		ckks_square(ctx, &ctx->aux[0], &ctx->aux[0]);
	}
	return;
}

__host__ cipher_t* ckks_power(
	CKKSContext *ctx,
	cipher_t *c1,
	uint64_t x){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_power(ctx, c2, c1, x);

	return c2;
}

// Horner's rule
__host__ void ckks_eval_polynomial(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *val,
	double *coeffs,
	int n){
	assert(CUDAEngine::is_init);

	assert(val != result);
	cipher_clear(ctx, result);
	result->slots = val->slots;

	for(int i = n-1; i >= 0; i--){
		ckks_mul(ctx, result, result, val);
		ckks_add(ctx, result, result, coeffs[i]);
	}

	return;
}

__host__ cipher_t* ckks_eval_polynomial(
	CKKSContext *ctx,
	cipher_t *x,
	double *coeffs,
	int n){
	assert(CUDAEngine::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_eval_polynomial(ctx, c2, x, coeffs, n);

	return c2;
}

__host__ void ckks_exp(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct){

	ckks_eval_polynomial(
		ctx,
		result,
		ct,
		ctx->maclaurin_coeffs[EXPONENT],
		8);

}

__host__ cipher_t* ckks_exp(
	CKKSContext *ctx,
	cipher_t *ct){
	assert(CUDAEngine::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_exp(ctx, result, ct);

	return result;
}

__host__ void ckks_sin(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct){

	ckks_eval_polynomial(
		ctx,
		result,
		ct,
		ctx->maclaurin_coeffs[SIN],
		12);

}

__host__ cipher_t* ckks_sin(
	CKKSContext *ctx,
	cipher_t *ct){
	assert(CUDAEngine::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_sin(ctx, result, ct);

	return result;
}

__host__ void ckks_cos(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct){

	ckks_eval_polynomial(
		ctx,
		result,
		ct,
		ctx->maclaurin_coeffs[COS],
		12);

}

__host__ cipher_t* ckks_cos(
	CKKSContext *ctx,
	cipher_t *ct){
	assert(CUDAEngine::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_cos(ctx, result, ct);

	return result;
}

__host__ void ckks_sigmoid(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct){

	ckks_eval_polynomial(
		ctx,
		result,
		ct,
		ctx->maclaurin_coeffs[SIGMOID],
		8);

}

__host__ cipher_t* ckks_sigmoid(
	CKKSContext *ctx,
	cipher_t *ct){
	assert(CUDAEngine::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_sigmoid(ctx, result, ct);

	return result;
}

__host__ void ckks_log1minus(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct){

	// ckks_eval_polynomial(
	// 	ctx,
	// 	result,
	// 	ct,
	// 	ctx->maclaurin_coeffs[LN1MINUS],
	// 	12);
	
	cipher_copy(ctx, result, ct);
	ckks_add(ctx, result, result, -1);
	ckks_mul(ctx, result, result, -1);
	ckks_log(ctx, result, result);

}

__host__ cipher_t* ckks_log1minus(
	CKKSContext *ctx,
	cipher_t *ct){
	assert(CUDAEngine::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_log1minus(ctx, result, ct);

	return result;
}

__host__ void ckks_rotate_right(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	int rotSlots){

	// Rotate_right
	rotate_slots_right(ctx, &c2->c[0], &c1->c[0], rotSlots);
	rotate_slots_right(ctx, &ctx->axax, &c1->c[1], rotSlots);

	// Key-switch
	// ModUp
	poly_modup(ctx, ctx->d2axaxQB, &ctx->axax, c1->level);

	// Multiply \tilde{d2} by the evk
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->rtk_right[rotSlots]->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->rtk_right[rotSlots]->a);

	// // ModDown
	poly_moddown(ctx, &ctx->d_tildeQ[0], &ctx->d_tildeQB[0], c1->level);
	poly_moddown(ctx, &c2->c[1], 		&ctx->d_tildeQB[1], c1->level);

	poly_add(ctx, &c2->c[0], &c2->c[0], &ctx->d_tildeQ[0]);

	c2->level = c1->level; 
	c2->slots = c1->slots; 
	c2->scale = c1->scale; 
}

__host__ void ckks_rotate_left(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	int rotSlots){

	// Rotate_left
	rotate_slots_left(ctx, &c2->c[0], &c1->c[0], rotSlots);
	rotate_slots_left(ctx, &ctx->axax, &c1->c[1], rotSlots);

	// Key-switch
	// ModUp
	poly_modup(ctx, ctx->d2axaxQB, &ctx->axax, c1->level);

	// Multiply \tilde{d2} by the evk
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->rtk_left[rotSlots]->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->rtk_left[rotSlots]->a);

	// // ModDown
	poly_moddown(ctx, &ctx->d_tildeQ[0], &ctx->d_tildeQB[0], c1->level);
	poly_moddown(ctx, &c2->c[1], 		&ctx->d_tildeQB[1], c1->level);

	poly_add(ctx, &c2->c[0], &c2->c[0], &ctx->d_tildeQ[0]);

	c2->level = c1->level; 
	c2->slots = c1->slots; 
	c2->scale = c1->scale; 
}

__host__ void ckks_conjugate(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1){

	// Conjugate
	poly_copy(ctx, &c2->c[0], &c1->c[0]);
	poly_copy(ctx, &ctx->axax, &c1->c[1]);

	conjugate_slots(ctx, &c2->c[0]);
	conjugate_slots(ctx, &ctx->axax);

	// Key-switch
	// ModUp
	poly_modup(ctx, ctx->d2axaxQB, &ctx->axax, c1->level);

	// Multiply \tilde{d2} by the cjk
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->cjk->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->cjk->a);

	// // ModDown
	poly_moddown(ctx, &ctx->d_tildeQ[0], &ctx->d_tildeQB[0], c1->level);
	poly_moddown(ctx, &c2->c[1], 		&ctx->d_tildeQB[1], c1->level);

	poly_add(ctx, &c2->c[0], &c2->c[0], &ctx->d_tildeQ[0]);

	c2->level = c1->level; 
	c2->slots = c1->slots; 
	c2->scale = c1->scale; 
}

__host__ cipher_t* ckks_conjugate(
	CKKSContext * ctx,
	cipher_t *ct){

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_conjugate(ctx, result, ct);

	return result;
}

// Receives "n" ciphertexts with 1 slot and merge in one ciphertext with n slots
__host__ void ckks_merge(
	CKKSContext * ctx,
	cipher_t *c_merged,
	cipher_t *cts,
	int n){

	assert(n <= CUDAEngine::N);
	cipher_clear(ctx, c_merged);
	c_merged->slots = 0;
	c_merged->scale = cts[0].scale;
	c_merged->level = cts[0].level;

	for (int i = 0; i < n; i++){
		ckks_discard_slots_except(ctx, &cts[i], 0);
		assert(c_merged->scale == cts[i].scale);
		assert(c_merged->level == cts[i].level);

		poly_double_add(
			ctx,
			&c_merged->c[0], &c_merged->c[0], &cts[i].c[0],
			&c_merged->c[1], &c_merged->c[1], &cts[i].c[1]);

		ckks_rotate_right(ctx, c_merged, c_merged, 1);
		c_merged->slots++;
	}
}

__host__ cipher_t* ckks_merge(
	CKKSContext * ctx,
	cipher_t *cts,
	int n){

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_merge(ctx, result, cts, n);

	return result;
}

__host__ void ckks_discard_higher_slots(
	CKKSContext * ctx,
	cipher_t *ct){
  
  	ckks_encrypt(ctx, &ctx->aux[0], 1, 1, ct->slots-1);
  	ckks_mul(ctx, ct, ct, &ctx->aux[0]);
}

__host__ void ckks_discard_slots_except(
	CKKSContext * ctx,
	cipher_t *ct,
	int idx){
  	
  	memset(ctx->h_val, 0, ct->slots * sizeof(complex<double>));
  	ctx->h_val[idx] = {1,0};

  	ckks_encrypt(ctx, &ctx->aux[0], ctx->h_val, ct->slots);
  	ckks_mul(ctx, ct, ct, &ctx->aux[0]);
}

__host__ void ckks_inverse(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct,
	int steps){

	
	ckks_add(ctx, &ctx->aux[0], ct, -1); // cbar
	ckks_mul(ctx, &ctx->aux[0], &ctx->aux[0], -1);

	ckks_add(ctx, result, &ctx->aux[0], 1); // tmp

	for (int i = 1; i < steps; i++){
		ckks_square(ctx, &ctx->aux[0], &ctx->aux[0]);
		ckks_add(ctx, &ctx->aux[1], &ctx->aux[0], 1);
		ckks_mul(ctx, result, &ctx->aux[1], result);
	}

}

__host__ cipher_t* ckks_inverse(
	CKKSContext *ctx,
	cipher_t *ct,
	int steps){
	assert(CUDAEngine::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_inverse(ctx, result, ct, steps);

	return result;
}

// approximation based on the area hyperbolic tangent function
// __host__ void ckks_log(
// 	CKKSContext *ctx,
// 	cipher_t *result,
// 	cipher_t *ct,
// 	int steps){

// 	// Compute the new evaluation point
// 	ckks_add(ctx, &ctx->aux[3], ct, -1);
// 	ckks_add(ctx, &ctx->aux[4], ct, 1);
// 	ckks_inverse(ctx, &ctx->aux[4], &ctx->aux[4], 5); // Uses aux 0, 1, and 2

// 	ckks_mul(ctx, &ctx->aux[0], &ctx->aux[3], &ctx->aux[4]); // (z-1)/(z+1)

// 	// Get the coefficients
// 	double *coeffs = ctx->maclaurin_coeffs[LOGARITHM];

// 	ckks_square(ctx, &ctx->aux[1], &ctx->aux[0]);

// 	// Eval the polynomial
// 	cipher_clear(ctx, result);
// 	result->slots = ct->slots;
// 	for(int i = 1; i < steps; i+=2){
// 		ckks_mul(ctx, &ctx->aux[2], &ctx->aux[0], coeffs[i]);
// 		ckks_add(ctx, result, result, &ctx->aux[2]);
// 		if(i <= steps - 1)
// 			ckks_mul(ctx, &ctx->aux[0], &ctx->aux[0], &ctx->aux[1]); // x^i * x^2
// 	}

// 	ckks_mul(ctx, result, result, 2);

// }

// A simple taylor series approximation
__host__ void ckks_log(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct,
	int steps){

	// Compute the new evaluation point
	ckks_add(ctx, &ctx->aux[0], ct, -1);
	cipher_copy(ctx, &ctx->aux[1], &ctx->aux[0]);

	// Eval the polynomial
	cipher_clear(ctx, result);
	result->slots = ct->slots;
	for(int i = 0; i < steps; i++){
		ckks_mul(ctx, &ctx->aux[2], &ctx->aux[1], 1.0 / (i+1));
		if(i % 2 == 0)
			ckks_add(ctx, result, result, &ctx->aux[2]);
		else
			ckks_sub(ctx, result, result, &ctx->aux[2]);
		if(i < steps - 1)
			ckks_mul(ctx, &ctx->aux[1], &ctx->aux[1], &ctx->aux[0]); //
	}

}


__host__ cipher_t* ckks_log(
	CKKSContext *ctx,
	cipher_t *ct,
	int steps){
	assert(CUDAEngine::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_log(ctx, result, ct, steps);

	return result;
}
