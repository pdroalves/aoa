#include <newckks/ckks/ckks.h>
#include <openssl/sha.h>


void print_hash(CKKSContext *ctx, poly_t *p, int i){
	std::string residue = poly_residue_to_string(ctx, p, i);
	unsigned char obuf[20];

	SHA1((unsigned char*)residue.c_str(), strlen(residue.c_str()), obuf);
	for (i = 0; i < 20; i++) 
    	printf("%02x ", obuf[i]);
    std::cout << std::endl;
}

void print_residue(CKKSContext *ctx, poly_t *p, int i){
    std::cout << poly_residue_to_string(ctx, p, i) << std::endl;
}

void print_decrypt_full(CKKSContext *cipher, cipher_t *c){
	cudaDeviceSynchronize();
	std::complex<double> *v = ckks_decrypt(cipher, c, cipher->keys.sk);
	cudaDeviceSynchronize();
	std::cout << "(" << c->slots << ") (level " << c->level << ") ";
	for(int i = 0; i < c->slots; i++)
		std::cout << (abs(real(v[i])) > 1e-10? real(v[i]) : 0) << ", ";
	std::cout << std::endl;
	free(v);
	cudaDeviceSynchronize();
}

void print_decrypt(CKKSContext *cipher, cipher_t *c, int n){
	cudaDeviceSynchronize();
	std::complex<double> *v = ckks_decrypt(cipher, c, cipher->keys.sk);
	cudaDeviceSynchronize();
	std::cout << "(" << c->slots << ") (level " << c->level << ") ";
	for(int i = 0; i < n; i++)
		std::cout << v[i] << ", ";
	std::cout << " ... ";
	for(int i = 0; i < n; i++)
		std::cout << v[c->slots - n - 1 + i] << ", ";
	std::cout << std::endl;
	free(v);
	cudaDeviceSynchronize();
}

__host__ CKKSKeychain* ckks_keygen(CKKSContext *ctx, SecretKey *sk){
	//////////////////////////////////
	// Alloc memory for each key //
	//////////////////////////////////
	CKKSKeychain *keys = new CKKSKeychain;

	////////////////
	// Public key //
	////////////////
	keys->pk = ckks_new_pk(ctx, sk);

	//////////////////////
	// Computation keys //
	//////////////////////

	// Evaluation key //
	keys->evk = ckks_new_mtk(ctx, sk);

	// Rotation key //
	keys->rtk_left = ckks_new_rtk_left(ctx, sk);
	keys->rtk_right = ckks_new_rtk_right(ctx, sk);

	// Conjugation key //
	keys->cjk = ckks_new_cjk(ctx, sk);

	//////////////////////
	ctx->keys = *keys;

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
	assert(CUDAManager::is_init);
	cipher_init(ctx, ct);

	ctx->get_sampler()->sample_Z0(ctx->u, QBase);
	ctx->get_sampler()->sample_DG(ctx->e, QBase);

	// ct = v * pk + (m + e1, e2)
	poly_add(ctx, ctx->e, ctx->e, m);
	poly_mul_add(ctx, &ct->c[0], ctx->u, &ctx->keys.pk->b, ctx->e);

	ctx->get_sampler()->sample_DG(ctx->e, QBase);
	poly_mul_add(ctx, &ct->c[1], ctx->u, &ctx->keys.pk->a, ctx->e);

	poly_clear(ctx, ctx->u);
	poly_clear(ctx, ctx->e);
	return ct;

}

__host__ cipher_t* ckks_encrypt(
	CKKSContext *ctx,
	std::complex<double>* val,
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
	std::complex<double>* val,
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

	std::complex<double> *cval = new std::complex<double>[slots];
	for (int i = 0; i < slots; i++)
		cval[i] = {val, 0};
	cipher_t *r = ckks_encrypt(ctx, cval, slots, empty_slots);
	delete[] cval;
	return r;
}

__host__ cipher_t* ckks_encrypt(
	CKKSContext *ctx,
	cipher_t *ct,
	double val,
	int slots,
	int empty_slots){

	std::complex<double> *cval = new std::complex<double>[slots];
	for (int i = 0; i < slots; i++)
		cval[i] = {val, 0};
	cipher_t *r = ckks_encrypt(ctx, ct, cval, slots, empty_slots);
	delete[] cval;
	return r;
}

__host__ poly_t* ckks_decrypt_poly(
	CKKSContext *ctx, 
	poly_t *m, 
	cipher_t *c, 
	SecretKey *sk){
	assert(CUDAManager::is_init);

	//////////////////////////////////////////
	// Compute x = |(c0 + c1*s)|_q //
	//////////////////////////////////////////
	poly_mul_add(ctx, m, &c->c[1], &sk->s, &c->c[0]);

	return m;
}


__host__ std::complex<double>* ckks_decrypt(
	CKKSContext *ctx,
	std::complex<double> *val,
	cipher_t *c,
	SecretKey *sk){
	assert(CUDAManager::is_init);

	ckks_decrypt_poly(ctx, ctx->m, c, sk);

	// Decode
	ctx->decode(val, ctx->m, c->scale, c->slots);		
	return val;
}

__host__ std::complex<double>* ckks_decrypt(
	CKKSContext *ctx,
	cipher_t *c,
	SecretKey *sk){
	assert(CUDAManager::is_init);

	std::complex<double> *val = new std::complex<double>[c->slots];

	return ckks_decrypt(ctx, val, c, sk);
}

__host__ void ckks_rescale(
	CKKSContext *ctx, 
	cipher_t *c){

	if(c->level == 0)
		throw std::runtime_error(
			"The ciphertext level is 0. No more rescaling is possible."
			);
	poly_rescale(ctx, &c->c[0], &c->c[1], c->level);
	c->level--;
}

cipher_t* ckks_add(CKKSContext *ctx, cipher_t *c1, cipher_t *c2){
	cipher_t *c3 = new cipher_t;

	ckks_add(ctx, c3, c1, c2);

	return c3;
}

__host__  void ckks_add(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2){
	assert(CUDAManager::is_init);
	assert(c1->slots == c2->slots);

	poly_add_add(ctx, &c3->c[0], &c1->c[0], &c2->c[0], &c3->c[1], &c1->c[1], &c2->c[1]);

	c3->level = std::min(c1->level, c2->level);
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
	assert(CUDAManager::is_init);
	assert(c1->slots == c2->slots);

	poly_sub(ctx, &c3->c[0], &c1->c[0], &c2->c[0]);
	poly_sub(ctx, &c3->c[1], &c1->c[1], &c2->c[1]);

	c3->level = std::min(c1->level, c2->level);
	c3->slots = c1->slots;
	c3->scale = c1->scale;
	return;
}

__host__ void ckks_mul_without_rescale(
	CKKSContext *ctx,
	cipher_t *c3,
	cipher_t *c1,
	cipher_t *c2){

	assert(CUDAManager::is_init);
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
	c3->level = std::min(c1->level, c2->level);
	////////////////////////////////////////////
	// Execute the homomorphic multiplication //
	////////////////////////////////////////////
	
	////////////////////////
	// Compute DR2(c1,c2) //
	////////////////////////
	poly_add_add(
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
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->keys.evk->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->keys.evk->a);

	// // ModDown
	poly_moddown(ctx, &ctx->d_tildeQ[0], &ctx->d_tildeQB[0], c3->level);
	poly_moddown(ctx, &ctx->d_tildeQ[1], &ctx->d_tildeQB[1], c3->level);

	// Output
	poly_add_add(ctx,
		&c3->c[0], &ctx->bxbx, &ctx->d_tildeQ[0],
		&c3->c[1], &ctx->axbx1, &ctx->d_tildeQ[1]);
	poly_sub(ctx, &c3->c[1], &c3->c[1], &ctx->bxbx);
	poly_sub(ctx, &c3->c[1], &c3->c[1], &ctx->axax);

	c3->slots = c1->slots; 
	c3->scale = NTL::conv<uint64_t>(
			NTL::to_ZZ(c1->scale) * NTL::to_ZZ(c2->scale) / CUDAManager::RNSQPrimes[c3->level]
		);
	return;
}

__host__ void ckks_mul(
	CKKSContext *ctx,
	cipher_t *c3,
	cipher_t *c1,
	cipher_t *c2){
	assert(CUDAManager::is_init);

	////////////////////////////////////////////
	// Execute the homomorphic multiplication //
	////////////////////////////////////////////
	ckks_mul_without_rescale(ctx, c3, c1, c2);
	ckks_rescale(ctx, c3);
	return;
}

__host__ cipher_t* ckks_mul(CKKSContext *ctx, cipher_t *c1, cipher_t *c2){
	assert(CUDAManager::is_init);

	cipher_t *c3 = new cipher_t;
	cipher_init(ctx, c3);

	ckks_mul(ctx, c3, c1, c2);

	return c3;
}

__host__ cipher_t* ckks_square(CKKSContext *ctx, cipher_t *c1){
	assert(CUDAManager::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_square(ctx, c2, c1);

	return c2;
}

__host__ void ckks_square_without_rescale(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1){
	assert(CUDAManager::is_init);
	assert(c1->c[0].state != NONINITIALIZED);
	c2->level = c1->level;
	
	////////////////////////
	// Compute DR2(c1,c2) //
	////////////////////////
	poly_add(ctx, &ctx->axbx1, &c1->c[1], &c1->c[0]);
	poly_mul(ctx, &ctx->axbx1, &ctx->axbx1, &ctx->axbx1); // d1
	poly_mul(ctx, &ctx->bxbx, &c1->c[0], &c1->c[0]); // d0
	poly_mul(ctx, &ctx->axax, &c1->c[1], &c1->c[1]); // d2

	// ModUp
	poly_modup(ctx, ctx->d2axaxQB, &ctx->axax, c2->level);

	// Multiply \tilde{d2} by the evk
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->keys.evk->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->keys.evk->a);

	// // ModDown
	poly_moddown(ctx, &ctx->d_tildeQ[0], &ctx->d_tildeQB[0], c2->level);
	poly_moddown(ctx, &ctx->d_tildeQ[1], &ctx->d_tildeQB[1], c2->level);

	// Output
	poly_add_add(ctx,
		&c2->c[0], &ctx->bxbx, &ctx->d_tildeQ[0],
		&c2->c[1], &ctx->axbx1, &ctx->d_tildeQ[1]);
	poly_sub(ctx, &c2->c[1], &c2->c[1], &ctx->bxbx);
	poly_sub(ctx, &c2->c[1], &c2->c[1], &ctx->axax);

	c2->slots = c1->slots; 
	c2->scale = NTL::conv<uint64_t>(
			NTL::to_ZZ(c1->scale) * NTL::to_ZZ(c1->scale) / CUDAManager::RNSQPrimes[c2->level]
		);
	return;
}

__host__ void ckks_square(CKKSContext *ctx, cipher_t *c2, cipher_t *c1){
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);
	assert(c1->slots == c2->slots);

	cipher_init(ctx, c3);
	c3->slots = c1->slots;
	for(int i = 0; i < size; i++){
		ckks_mul_without_rescale(ctx, &ctx->caux[0], &c1[i], &c2[i]);
		ckks_add(ctx, c3, c3, &ctx->caux[0]);
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
	assert(CUDAManager::is_init);

	cipher_init(ctx, c3);
	c3->slots = c1->slots;
	for(int i = 0; i < size; i++){
		ckks_mul_without_rescale(ctx, &ctx->caux[0], &c1[i], c2[i]);
		ckks_add(ctx, c3, c3, &ctx->caux[0]);
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
	assert(CUDAManager::is_init);

	ckks_mul(ctx, c3, c1, c2);
	ckks_sumslots(ctx, c3, c3);
	return;
}

template __host__ void ckks_batch_inner_prod<cipher_t>(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);
template __host__ void ckks_batch_inner_prod<std::complex<double>>(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, std::complex<double> *c2);
template __host__ void ckks_batch_inner_prod<poly_t>(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, poly_t *c2);

__host__ void ckks_sumslots(CKKSContext *ctx, cipher_t *c2, cipher_t *c1){
	// Rotate and add to sum slots
	if(c1 != c2)
		cipher_copy(ctx, c2, c1);
	for(int i = (c2->slots >> 1); i >= 1; i /= 2){
		ckks_rotate_left(ctx, &ctx->caux[0], c2, i);
		// std::cout << "c2: " << std::endl;
		// print_decrypt(ctx, c2, 2);
		// std::cout << "caux (rotation by " << i << "):"  << std::endl;
		// print_decrypt(ctx, &ctx->caux[0], 2);
		
		ckks_add(ctx, c2, c2, &ctx->caux[0]);
		// std::cout << "c2 + caux: " << std::endl;
		// print_decrypt(ctx, c2, 2);
		// std::cout << std::endl;
	}
}

__host__ cipher_t* ckks_sumslots(
	CKKSContext *ctx,
	cipher_t *c){
	assert(CUDAManager::is_init);

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
	
	assert(CUDAManager::is_init);
	uint64_t cnst = abs(val) * c1->scale;

	if(val >= 0)
		poly_add(ctx, &c2->c[0], &c1->c[0], cnst);
	else
		poly_sub(ctx, &c2->c[0], &c1->c[0], cnst);

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
	assert(CUDAManager::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_add(ctx, c2, c1, val);

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
	assert(CUDAManager::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_mul(ctx, c2, c1, val);

	return c2;
}

__host__ void ckks_mul_without_rescale(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	double val){
	assert(CUDAManager::is_init);

	uint64_t x = lrint(abs(val) * (c1->scale));

	poly_mul(ctx, &c2->c[0], &c1->c[0], x);
	poly_mul(ctx, &c2->c[1], &c1->c[1], x);

	if(val < 0){
		poly_negate(ctx, &c2->c[0]);
		poly_negate(ctx, &c2->c[1]);
	}
	c2->level = c1->level;
	c2->slots = c1->slots;
	c2->scale = NTL::conv<uint64_t>(
		NTL::to_ZZ(c1->scale) * NTL::to_ZZ(c1->scale) / CUDAManager::RNSQPrimes[c1->level]
		);

	return;
}

__host__ cipher_t* ckks_mul_without_rescale(
	CKKSContext *ctx,
	cipher_t *c1,
	double val){
	assert(CUDAManager::is_init);

	cipher_t *c2 = new cipher_t;
	cipher_init(ctx, c2);

	ckks_mul_without_rescale(ctx, c2, c1, val);

	return c2;
}

__host__ void ckks_mul(
	CKKSContext *ctx,
	cipher_t *c2,
	cipher_t *c1,
	std::complex<double> *val){
	assert(CUDAManager::is_init);

	// Encode
	ctx->encode(ctx->m, &c1->scale, val, c1->slots);
	assert(ctx->m->state != NONINITIALIZED);

	poly_mul(ctx, &c2->c[0], &c1->c[0], ctx->m);
	poly_mul(ctx, &c2->c[1], &c1->c[1], ctx->m);

	c2->level = c1->level;
	c2->slots = c1->slots;
	c2->scale = NTL::conv<uint64_t>(
		NTL::to_ZZ(c1->scale) * NTL::to_ZZ(c1->scale) / CUDAManager::RNSQPrimes[c1->level]
		);

	ckks_rescale(ctx, c2);
	return;
}

__host__ cipher_t* ckks_mul(
	CKKSContext *ctx,
	cipher_t *c1,
	std::complex<double> *val){
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);

	poly_mul(ctx, &c2->c[0], &c1->c[0], val);
	poly_mul(ctx, &c2->c[1], &c1->c[1], val);

	c2->level = c1->level;
	c2->slots = c1->slots;
	c2->scale = NTL::conv<uint64_t>(
		NTL::to_ZZ(c1->scale) * NTL::to_ZZ(c1->scale) / CUDAManager::RNSQPrimes[c1->level]
		);

	return;
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
	assert(CUDAManager::is_init);

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

	assert(CUDAManager::is_init);
	cipher_init(ctx, c2);
	if(x == 0){
		ckks_encrypt(ctx, c2, 1, c1->slots);
		return;
	}
	cipher_clear(ctx, &ctx->caux[0]);
	cipher_copy(ctx, &ctx->caux[0], c1);
	while(x > 0){
		ckks_square(ctx, &ctx->caux[0], &ctx->caux[0]);
		x >>= 1;
	}
	return;
}

__host__ cipher_t* ckks_power_of_2(
	CKKSContext *ctx,
	cipher_t *c1,
	uint64_t x){
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);
	cipher_init(ctx, c2, c1->slots);

	if(x == 0){
		ckks_encrypt(ctx, c2, 1, c1->slots);
		return;
	}

	cipher_clear(ctx, &ctx->caux[0]);
	cipher_copy(ctx, &ctx->caux[0], c1);
	bool first_op = true;
	while(x > 0){
		if(x % 2 == 1){
			if(first_op){
				ckks_add(ctx, c2, c2, &ctx->caux[0]);
				first_op = false;
			}
			else
				ckks_mul(ctx, c2, c2, &ctx->caux[0]);
		}
		x >>= 1;
		ckks_square(ctx, &ctx->caux[0], &ctx->caux[0]);
	}
	return;
}

__host__ cipher_t* ckks_power(
	CKKSContext *ctx,
	cipher_t *c1,
	uint64_t x){
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);

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
		11);

}

__host__ cipher_t* ckks_exp(
	CKKSContext *ctx,
	cipher_t *ct){
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);

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
	assert(CUDAManager::is_init);

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
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->keys.rtk_right[rotSlots]->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->keys.rtk_right[rotSlots]->a);

	// ModDown
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
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->keys.rtk_left[rotSlots]->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->keys.rtk_left[rotSlots]->a);

	// ModDown
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
	poly_mul(ctx, &ctx->d_tildeQB[0], ctx->d2axaxQB, &ctx->keys.cjk->b);
	poly_mul(ctx, &ctx->d_tildeQB[1], ctx->d2axaxQB, &ctx->keys.cjk->a);

	// ModDown
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

	assert(n <= CUDAManager::N);
	cipher_clear(ctx, c_merged);
	c_merged->slots = 0;
	c_merged->scale = cts[0].scale;
	c_merged->level = cts[0].level;

	for (int i = 0; i < n; i++){
		ckks_discard_slots_except(ctx, &cts[i], 0);
		assert(c_merged->scale == cts[i].scale);
		assert(c_merged->level == cts[i].level);

		poly_add_add(
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
  
  	ckks_encrypt(ctx, &ctx->caux[0], 1, 1, ct->slots-1);
  	ckks_mul(ctx, ct, ct, &ctx->caux[0]);
}

__host__ void ckks_discard_slots_except(
	CKKSContext * ctx,
	cipher_t *ct,
	int idx){
	assert(idx < ct->slots);
  	
  	memset(ctx->h_val, 0, ct->slots * sizeof(std::complex<double>));
  	ctx->h_val[idx] = {1,0};

	ckks_encrypt(ctx, &ctx->caux[0], ctx->h_val, ct->slots);
	ckks_mul(ctx, ct, ct, &ctx->caux[0]);
}

__host__ void ckks_inverse(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct,
	int steps){

	
	ckks_add(ctx, &ctx->caux[0], ct, -1); // cbar
	ckks_mul(ctx, &ctx->caux[0], &ctx->caux[0], -1);

	ckks_add(ctx, result, &ctx->caux[0], 1); // tmp

	for (int i = 1; i < steps; i++){
		ckks_square(ctx, &ctx->caux[0], &ctx->caux[0]);
		ckks_add(ctx, &ctx->caux[1], &ctx->caux[0], 1);
		ckks_mul(ctx, result, &ctx->caux[1], result);
	}

}

__host__ cipher_t* ckks_inverse(
	CKKSContext *ctx,
	cipher_t *ct,
	int steps){
	assert(CUDAManager::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_inverse(ctx, result, ct, steps);

	return result;
}

// A simple taylor series approximation
__host__ void ckks_log(
	CKKSContext *ctx,
	cipher_t *result,
	cipher_t *ct,
	int steps){

	// Compute the new evaluation point
	ckks_add(ctx, &ctx->caux[0], ct, -1);
	cipher_copy(ctx, &ctx->caux[1], &ctx->caux[0]);

	// Eval the polynomial
	cipher_clear(ctx, result);
	result->slots = ct->slots;
	for(int i = 0; i < steps; i++){
		ckks_mul(ctx, &ctx->caux[2], &ctx->caux[1], 1.0 / (i+1));
		if(i % 2 == 0)
			ckks_add(ctx, result, result, &ctx->caux[2]);
		else
			ckks_sub(ctx, result, result, &ctx->caux[2]);
		if(i < steps - 1)
			ckks_mul(ctx, &ctx->caux[1], &ctx->caux[1], &ctx->caux[0]); //
	}

}


__host__ cipher_t* ckks_log(
	CKKSContext *ctx,
	cipher_t *ct,
	int steps){
	assert(CUDAManager::is_init);

	cipher_t *result = new cipher_t;
	cipher_init(ctx, result);

	ckks_log(ctx, result, ct, steps);

	return result;
}
