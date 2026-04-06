#include <AOADGT/cuda/sampler.h>

curandGenerator_t Sampler::gen;
curandState *Sampler::states;

/**
 * @brief       Set each generator
 * 
 * @param states [description]
 * @param seed   [description]
 * @param N   [Number of states that must be initialized]
 */
__global__ void setup ( 
		curandState * states, 
		unsigned long seed,
		const int size ){
		
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < size){
		// We use seed + tid as seed of the tid-th generator. This way we garantee
		// that each one will be initiated with a different seed.
		curand_init ( seed, tid, 0, &states[tid] ); // replace by curand()?
		curand(&states[tid]); // Discard the first result	
	}
}

__host__ void call_setup(Context *ctx, curandState *states){
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(QBBase) > 0);

	int nresidues = CUDAEngine::get_n_residues(QBBase);
	const int ADDGRIDXDIM = (
		(CUDAEngine::N * nresidues)%ADDBLOCKXDIM == 0?
		(CUDAEngine::N * nresidues)/ADDBLOCKXDIM :
		(CUDAEngine::N * nresidues)/ADDBLOCKXDIM + 1
		);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	setup<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		states,
		SEED,
		CUDAEngine::N *nresidues);
	cudaCheckError();
}

/**
 * @brief       Sample a polynomial from a narrow distribution,
 * 
 * @param[out] coefs     An array of coefficients composed by concatenated residues.
 * @param[in]  N         Number of elements to be sampled for each residue.
 * @param[in]  nresidues Number of residues.
 * @param[in]  states    cuRand generators.
 */
__global__ void sample_from_narrow(
		GaussianInteger *coefs,
		uint64_t *coprimes,
		int N,
		int nresidues,
		curandState *states) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
 	const int cid = tid % N;

	if (tid < N){	
		/* Copy state to local memory for efficiency */
		curandState localState = states[cid];
		int64_t value1 = (curand(&localState) % 3);
		int64_t value2 = (curand(&localState) % 3);

		for(int rid = 0; rid < nresidues; rid++){
			GaussianInteger xi;
			xi.re =  (value1 <= 1 ? value1 : coprimes[rid] - 1);
			xi.imag =  (value2 <= 1 ? value2 : coprimes[rid] - 1);

			coefs[cid + rid * N] = xi;
		}

		/* Copy state back to global memory */
		states[cid] = localState;
	}
				
}

/**
 * @brief       Sampling of a narrow polynomial
 * 
 * @param[out] p   		The outcome
 * @param[in]  ctx 		The context that shall be used
 */
__host__	void call_get_narrow_sample(
	poly_t *p,
	Context *ctx){
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(p->base) > 0);

	// Kernel configuration
	const int ADDGRIDXDIM = (
		(CUDAEngine::N)%ADDBLOCKXDIM == 0?
		(CUDAEngine::N)/ADDBLOCKXDIM :
		(CUDAEngine::N)/ADDBLOCKXDIM + 1
		);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	assert(Sampler::states != NULL && CUDAEngine::N > 0);

	/** 
	 * Generate values
	 */
	sample_from_narrow<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		p->d_coefs,
		CUDAEngine::RNSCoprimes,
		CUDAEngine::N,
		CUDAEngine::get_n_residues(p->base),
		Sampler::states);
	cudaCheckError();

	DGTEngine::execute_dgt( ctx, p, FORWARD);  

}

/**
 * @brief       Sample a polynomial from a binary distribution,
 * 
 * @param[out] coefs     An array of coefficients composed by concatenated residues.
 * @param[in]  N         Number of elements to be sampled for each residue.
 * @param[in]  nresidues Number of residues.
 * @param[in]  states    cuRand generators.
 */
__global__ void sample_from_binary(
		GaussianInteger *coefs,
		uint64_t *coprimes,
		int N,
		int nresidues,
		curandState *states) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
 	const int cid = tid % N;

	if (tid < N){	
		/* Copy state to local memory for efficiency */
		curandState localState = states[cid];
		int value1 = (curand(&localState) % 2);
		int value2 = (curand(&localState) % 2);

		for(int rid = 0; rid < nresidues; rid++){
			GaussianInteger xi = (GaussianInteger){value1, value2};

			coefs[cid + rid * N] = xi;
		}

		/* Copy state back to global memory */
		states[cid] = localState;
	}
				
}

/**
 * @brief       Sampling of a binary polynomial
 * 
 * @param[out] p   		The outcome
 * @param[in]  ctx 		The context that shall be used
 */
__host__	void call_get_binary_sample(
	poly_t *p,
	Context *ctx){
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(p->base) > 0);

	// Kernel configuration
	const int ADDGRIDXDIM = (
		(CUDAEngine::N)%ADDBLOCKXDIM == 0?
		(CUDAEngine::N)/ADDBLOCKXDIM :
		(CUDAEngine::N)/ADDBLOCKXDIM + 1
		);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	assert(Sampler::states != NULL && CUDAEngine::N > 0);

	/** 
	 * Generate values
	 */
	sample_from_binary<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		p->d_coefs,
		CUDAEngine::RNSCoprimes,
		CUDAEngine::N,
		CUDAEngine::get_n_residues(p->base),
		Sampler::states);
	cudaCheckError();

	DGTEngine::execute_dgt(ctx, p, FORWARD);

}

/**
 * @brief       Sample from {-1, 0, 1} with probability 1/4 for -1 and +1 and 1/2 for 0
 * 
 * @param[out] p   		The outcome
 * @param[in]  ctx 		The context that shall be used
 */
__host__	void call_get_zo_sample(
	poly_t *p,
	Context *ctx){
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(p->base) > 0);

	/** 
	 * Generate values
	 */
	memset(ctx->h_aux, 0, poly_get_size(p->base));
	for(int j = 0; j < CUDAEngine::N; j++)
		ctx->h_aux[j] = (uint64_t)((rand() % 2) + 1);

	/**
	 * Shuffle
	 */
	shuffle(
		ctx->h_aux,
		ctx->h_aux + 2 * CUDAEngine::N,
		std::default_random_engine(SEED));

	/**
	 * Adjust negatives and convert to GaussianIntegers
	 */
	for(int i = 0; i < CUDAEngine::N; i++)
		for(int j = 0; j < CUDAEngine::get_n_residues(p->base); j++){
			ctx->h_coefs[i + j * CUDAEngine::N].re = (
				ctx->h_aux[i] > 1? 
				COPRIMES_BUCKET[j] - 1 : ctx->h_aux[i]);
			ctx->h_coefs[i + j * CUDAEngine::N].imag = (
				ctx->h_aux[i + CUDAEngine::N] > 1? 
				COPRIMES_BUCKET[j] - 1 : ctx->h_aux[i + CUDAEngine::N]);
		}

	cudaMemcpyAsync(
		p->d_coefs,
		ctx->h_coefs,
		poly_get_size(p->base),
		cudaMemcpyHostToDevice,
		ctx->get_stream());
	DGTEngine::execute_dgt(ctx, p, FORWARD);

}


/**
 * @brief       Sample a polynomial from a binary distribution,
 * 
 * @param[out] coefs     An array of coefficients composed by concatenated residues.
 * @param[in]  N         Number of elements to be sampled for each residue.
 * @param[in]  nresidues Number of residues.
 * @param[in]  states    cuRand generators.
 * @param[in]  x 		Upper bound for each coefficient.
 */
__global__ void sample_from_uniform(
		GaussianInteger *coefs,	
		uint64_t *coprimes,
		int N,
		int nresidues,
		curandState *states) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
 	const int cid = tid % N ;

 	const int rid = tid / N ;

	if (tid < N * nresidues){	
		/* Copy state to local memory for efficiency */
		const uint64_t uppbound = coprimes[rid] - 1; 
		curandState localState = states[tid];

		GaussianInteger xi;
		xi.re   = (uint64_t) (curand_uniform_double(&localState) * uppbound);
		xi.imag = (uint64_t) (curand_uniform_double(&localState) * uppbound);

		coefs[cid + rid * N] = xi;
		
		/* Copy state back to global memory */
		states[tid] = localState;
	}
}

/**
 * @brief       Sampling of a binary polynomial in R_x
 * 
 * @param[out] p   		The outcome
 * @param[in]  x 		Upper bound for each coefficient.
 * @param[in]  ctx 		The context that shall be used
 */
__host__	void call_get_uniform_sample(
	poly_t *p,
	Context *ctx){
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(p->base) > 0);

	// Kernel configuration
	int nresidues = CUDAEngine::get_n_residues(p->base);
	const int ADDGRIDXDIM = (
		(CUDAEngine::N * nresidues)%ADDBLOCKXDIM == 0?
		(CUDAEngine::N * nresidues)/ADDBLOCKXDIM :
		(CUDAEngine::N * nresidues)/ADDBLOCKXDIM + 1
		);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	assert(Sampler::states != NULL && CUDAEngine::N > 0);

	/** 
	 * Generate values
	 */
	sample_from_uniform<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		p->d_coefs,
		CUDAEngine::RNSCoprimes,
		CUDAEngine::N,
		nresidues,
		Sampler::states);
	cudaCheckError();

	p->state = TRANSSTATE;
}

#ifndef min

 #define min(a,b) (((a) < (b)) ? (a) : (b))

#endif

/**
 * @brief       Sampling of a binary polynomial with Hamming weight h = HAMMINGWEIGHT_H
 * 
 * @param[out] p   		The outcome
 * @param[in]  ctx 		The context that shall be used
 */
__host__	void call_get_binary_hweight_sample(
	poly_t *p,
	Context *ctx){
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(p->base) > 0);

	/** 
	 * Generate values
	 */
	memset(ctx->h_aux, 0, poly_get_size(p->base));
	for(int j = 0; j < min(HAMMINGWEIGHT_H, 2 * CUDAEngine::N); j++)
		ctx->h_aux[j] = (uint64_t)((rand() % 2) + 1);

	/**
	 * Shuffle
	 */
	shuffle(
		ctx->h_aux,
		ctx->h_aux + 2 * CUDAEngine::N,
		std::default_random_engine(SEED));

	/**
	 * Adjust negatives and convert to GaussianIntegers
	 */
	for(int i = 0; i < CUDAEngine::N; i++)
		for(int j = 0; j < CUDAEngine::get_n_residues(p->base); j++){
			ctx->h_coefs[i + j * CUDAEngine::N].re = (
				ctx->h_aux[i] > 1? 
				COPRIMES_BUCKET[j] - 1 : ctx->h_aux[i]);
			ctx->h_coefs[i + j * CUDAEngine::N].imag = (
				ctx->h_aux[i + CUDAEngine::N] > 1? 
				COPRIMES_BUCKET[j] - 1 : ctx->h_aux[i + CUDAEngine::N]);
		}

	cudaMemcpyAsync(
		p->d_coefs,
		ctx->h_coefs,
		poly_get_size(p->base),
		cudaMemcpyHostToDevice,
		ctx->get_stream());
	DGTEngine::execute_dgt(ctx, p, FORWARD);

}

/**
 * @brief       Sample a polynomial from a discrete gaussian distribution
 * 
 * @param[out] coefs     An array of coefficients composed by concatenated residues.
 * @param[in]  N         Number of elements to be sampled for each residue.
 * @param[in]  nresidues Number of residues.
 * @param[in]  states    cuRand generators.
 */
__global__ void sample_from_discrete_gaussian( 
	GaussianInteger *coefs,
	uint64_t *coprimes,
	int N,
	int nresidues,
	curandState *states) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int cid = tid % N;

	if (tid < N){ 
		int64_t value1 = (int64_t)nearbyint(
				curand_normal_double(&states[cid])
				* GAUSSIAN_STD_DEVIATION + GAUSSIAN_BOUND
			);
		int64_t value2 = (int64_t)nearbyint(
				curand_normal_double(&states[cid])
				* GAUSSIAN_STD_DEVIATION + GAUSSIAN_BOUND
			);


		for(int rid = 0; rid < nresidues; rid++){
			GaussianInteger xi;
			xi.re =  (value1 >= 0 ? value1 : coprimes[rid] + value1);
			xi.imag =  (value2 >= 0 ? value2 : coprimes[rid] + value2);

			coefs[cid + rid * N] = xi;
		}
	}
}

/**
 * @brief       Sample a polynomial from a discrete gaussian distribution

 * @param[out] p   		The outcome
 * @param[in]  ctx 		The context that shall be used
 */
__host__ void call_get_normal_sample(	
		poly_t *p,
		Context *ctx){
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(p->base) > 0);

	// Kernel configuration
	const int ADDGRIDXDIM = (
		(CUDAEngine::N)%ADDBLOCKXDIM == 0?
		(CUDAEngine::N)/ADDBLOCKXDIM :
		(CUDAEngine::N)/ADDBLOCKXDIM + 1
		);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	/** 
	 * Generate values
	 */
	sample_from_discrete_gaussian<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		p->d_coefs,
		CUDAEngine::RNSCoprimes,
		CUDAEngine::N,
		CUDAEngine::get_n_residues(p->base),
		Sampler::states);
	cudaCheckError();

	DGTEngine::execute_dgt(ctx, p, FORWARD);

}


__host__ void Sampler::reset(Context *ctx){
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(QBBase) > 0);

	cudaStreamSynchronize(ctx->get_stream());
	destroy();
	init(ctx);
	cudaStreamSynchronize(ctx->get_stream());
}


__host__ void Sampler::init(Context *ctx){
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(CUDAEngine::get_n_residues(QBBase) > 0);

	curandStatus_t resultRand = curandCreateGenerator(
		&gen, 
		CURAND_RNG_PSEUDO_DEFAULT);
	assert(resultRand == CURAND_STATUS_SUCCESS);

	resultRand = curandSetPseudoRandomGeneratorSeed(
		gen, 
		SEED);
	assert(resultRand == CURAND_STATUS_SUCCESS);

	/**
	 * Setup
	*/
	const int size = CUDAEngine::N * CUDAEngine::get_n_residues(QBBase);
	assert(size > 0);
	cudaMalloc((void**)&states, size * sizeof(curandState));
	cudaCheckError();

	assert(states != NULL);		
	call_setup(ctx, states);
}

__host__ void Sampler::sample(
	Context *ctx,
	poly_t *p,
	int kind){

	assert(kind < KINDS_COUNT);
	poly_init(ctx, p);
	curandSetStream(gen, ctx->get_stream());

	const int size = CUDAEngine::N * CUDAEngine::get_n_residues(QBBase);

	switch(kind){
		case DISCRETE_GAUSSIAN:
			// Sample from a discrete gaussian distribution set on the constructor
			// 
			call_get_normal_sample(
				p,
				ctx);
			break;
		case NARROW:
			// Sample uniformly from {-1, 0, 1}
			// 
			call_get_narrow_sample(
				p,
				ctx);
			break;
		case ZO:
			// Sample from {-1, 0, 1} with probability 1/4 for -1 and +1 and 1/2 for 0
			// 
			call_get_zo_sample(
				p,
				ctx);
			break;
		case BINARY:
			// Sample uniformly from {0, 1}
			// 
			call_get_binary_sample(
				p,
				ctx);
			break;
		case HAMMINGWEIGHT:
			// Sample binaries from a Hamming weight H
			// 
			call_get_binary_hweight_sample(
				p,
				ctx);
			break;
		case UNIFORM:
			// Sample uniformly in q
			// 
			call_get_uniform_sample(
				p,
				ctx);
			break;
		default:
			throw std::runtime_error("Invalid sampler: " + std::to_string(kind));
	}
	// std::cout << kind << ") Sampled with |v|_infty = " << poly_infty_norm(*p) << std::endl;
}

__host__ poly_t* Sampler::sample( 
	Context *ctx,
	int kind){

	poly_t *p = new poly_t;

	sample(ctx, p, kind);
	return p;
}
