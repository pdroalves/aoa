#include <newckks/cuda/sampler.h>
#include <newckks/coprimes.h>

#define SEED (unsigned long long)(42)// We derandomize for debug purposes
#define HAMMINGWEIGHT (int)64

extern __constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];

/////////////
// Uniform //
/////////////
///
__global__ void uniform_cast(uint64_t *b, uint64_t *a, const int N, const int NResidues){
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int rid = tid / N;

	if( tid < N * NResidues )
		b[tid] = a[tid] % d_RNSCoprimes[rid];
}

__host__ void Sampler::sample_uniform(poly_t *p, poly_bases b){
	poly_init(ctx, p, b);

	curandSetStream(gen, ctx->get_stream());

	// This will generate 32-bits coefficients. It's a lazy way to assert
	// that all coefficients will be in the supported range.
	uint64_t size = CUDAManager::N * CUDAManager::get_n_residues(b);
	curandCheckError(
		curandGenerateLongLong(
			gen,
			(unsigned long long*)p->d_coefs,
			size)
		);

	const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
	const dim3 blockDim(DEFAULTBLOCKSIZE);
	uniform_cast<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		p->d_coefs,
		p->d_coefs,
		CUDAManager::N,
		CUDAManager::get_n_residues(b));
	cudaCheckError();

	p->state = RNSSTATE;
}

//////////////////////////////////
// Discrete (truncate) Gaussian //
//////////////////////////////////

__global__ void dg_cast(uint64_t *b, double *a, const int N, const int NResidues){
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int cid = tid % N;
	const int rid = tid / N;

	if( tid < N * NResidues ){
		int64_t vcast = (int64_t)(a[cid]);
		vcast = (vcast >= 0 ? vcast : d_RNSCoprimes[rid] + vcast);
		b[tid] = vcast;
	}
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
	uint64_t *coefs,
	int N, int nresidues,
	curandState *states) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int cid = tid % N;

	if (tid < N){ 
		curandState localState = states[tid];
		int64_t v = (int64_t)nearbyint(
				curand_normal_double(&localState)
				* GAUSSIAN_STD_DEVIATION + GAUSSIAN_BOUND
			);

		for(int rid = 0; rid < nresidues; rid++)
			coefs[cid + rid * N] =  (v >= 0 ? v : d_RNSCoprimes[rid] + v);

		states[tid] = localState;
	}
}

__host__ void Sampler::sample_DG(poly_t *p, poly_bases b){
	poly_init(ctx, p, b);

	uint64_t size = CUDAManager::N;
	const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
	const dim3 blockDim(DEFAULTBLOCKSIZE);

	/** 
	 * Generate values
	 */
	sample_from_discrete_gaussian<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		p->d_coefs,
		CUDAManager::N,
		CUDAManager::get_n_residues(b),
		this->states);
	cudaCheckError();

	p->state = RNSSTATE;

}

////////
// Z0 //
////////

__host__ void Sampler::sample_Z0(poly_t *p, poly_bases b){
	poly_init(ctx, p, b);

	/** 
	 * Generate values
	 */
	memset(h_tmp, 0, poly_get_size(b));
	for(int j = 0; j < (CUDAManager::N>>1); j++)
		h_tmp[j] = (uint64_t)((rand() % 2) + 1);

	/**
	 * Shuffle
	 */
	shuffle(
		h_tmp,
		h_tmp + CUDAManager::N,
		std::default_random_engine(SEED));

	/**
	 * Adjust negatives
	 */
	// #pragma omp parallel for
	for(int i = 0; i < CUDAManager::N; i++)
		for(int j = 0; j < CUDAManager::get_n_residues(b); j++)
			h_tmp[i + j * CUDAManager::N] = (
				h_tmp[i] > 1? 
				COPRIMES_BUCKET[j] - 1 : h_tmp[i]);

	cudaMemcpyAsync(
		p->d_coefs,
		h_tmp,
		poly_get_size(b),
		cudaMemcpyHostToDevice,
		ctx->get_stream());

	p->state = RNSSTATE;
}

__host__ void Sampler::sample_hw(poly_t *p, poly_bases b){
	poly_init(ctx, p, b);

	/** 
	 * Generate values
	 */
	memset(h_tmp, 0, CUDAManager::N * sizeof(uint64_t));
	for(int i = 0; i < min(HAMMINGWEIGHT, CUDAManager::N); i++)
		h_tmp[i] = (uint64_t)((rand() % 2) + 1);

	/**
	 * Shuffle
	 */
	shuffle(
		h_tmp,
		h_tmp + CUDAManager::N,
		std::default_random_engine(SEED));

	/**
	 * Adjust negatives
	 */
	for(int i = 0; i < CUDAManager::N; i++)
		for(int j = 0; j < CUDAManager::get_n_residues(b); j++)
			h_tmp[i + j * CUDAManager::N] = (
				h_tmp[i] > 1? 
				COPRIMES_BUCKET[j] - 1 : h_tmp[i]);

	cudaMemcpyAsync(
		p->d_coefs,
		h_tmp,
		CUDAManager::N * CUDAManager::get_n_residues(b) * sizeof(uint64_t),
		cudaMemcpyHostToDevice,
		ctx->get_stream());

	p->state = RNSSTATE;
}

__host__ void Sampler::sample_narrow(poly_t *p, poly_bases b){
	poly_init(ctx, p, b);

	/** 
	 * Generate values
	 */
	memset(h_tmp, 0, poly_get_size(b));
	for(int j = 0; j < CUDAManager::N; j++)
		h_tmp[j] = (uint64_t)((rand() % 3));

	/**
	 * Adjust negatives
	 */
	for(int i = 0; i < CUDAManager::N; i++)
		for(int j = 0; j < CUDAManager::get_n_residues(b); j++)
			h_tmp[i + j * CUDAManager::N] = (
				h_tmp[i] == 2? 
				COPRIMES_BUCKET[j] - 1 : h_tmp[i]);

	cudaMemcpyAsync(
		p->d_coefs,
		h_tmp,
		poly_get_size(b),
		cudaMemcpyHostToDevice,
		ctx->get_stream());

	p->state = RNSSTATE;
}

/**
 * @brief       Set each generator
 * 
 * @param states [description]
 * @param seed   [description]
 * @param N   [Number of states that must be initialized]
 */
__global__ void setup (curandState *states, const int size){
		
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < size)
		// We use seed + tid as seed of the tid-th generator. This way we garantee
		// that each one will be initiated with a different seed.
		curand_init ( SEED, tid, 0, &states[tid] ); // replace by curand()?
}

Sampler::Sampler(Context *ctx){
	assert(CUDAManager::is_init);

	curandCreateGenerator(
		&gen, 
		CURAND_RNG_QUASI_SOBOL64);

	// This is probably a bad seed choice for many reasons. 
	// One of the reasons is that every time someone runs Sampler::init(),
	// will initialize it with the exactly same seed.
	curandSetPseudoRandomGeneratorSeed(gen, SEED);

	if(hasSupportStreamAlloc())
		cudaMallocAsync((void**)&d_tmp, poly_get_size(QBBase), ctx->get_stream());
	else
		cudaMalloc((void**)&d_tmp, poly_get_size(QBBase));
	cudaCheckError();
	h_tmp = (uint64_t*) malloc (poly_get_size(QBBase));

	this->ctx = ctx;

	uint64_t size = CUDAManager::N * CUDAManager::get_n_residues(QBBase);
	const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
	const dim3 blockDim(DEFAULTBLOCKSIZE);
	assert(size > 0);
	cudaMalloc((void**)&states, size * sizeof(curandState));
	cudaCheckError();

	setup<<<gridDim, blockDim, 0, ctx->get_stream()>>>(
		states, size);

	is_init = true;
};

Sampler::~Sampler(){
	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();
	curandDestroyGenerator(gen);
	cudaCheckError();
	if(hasSupportStreamAlloc())
		cudaFreeAsync(d_tmp, ctx->get_stream());
	else
		cudaFree(d_tmp);
	cudaCheckError();
	free(h_tmp);

	cudaFree(states);
	cudaCheckError();
		
	is_init = false;
};