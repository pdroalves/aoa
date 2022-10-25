#include <newckks/ckks/encoder.h>
#include <newckks/cuda/htrans/ntt.h>
#include <newckks/coprimes.h>

NTL_CLIENT
extern __constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];

////////////////////
// FFT intrinsics //
////////////////////
/// This must be replaced by something of our own
void arrayBitReverse(std::complex<double>* vals, const long size) {
	for (long i = 1, j = 0; i < size; ++i) {
		long bit = size >> 1;
		for (; j >= bit; bit >>= 1) {
			j -= bit;
		}
		j += bit;
		if (i < j) {
			swap(vals[i], vals[j]);
		}
	}
}

__host__ __device__ uint32_t bitReverse(uint32_t x) {
	x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
	x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
	x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
	x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
	return ((x >> 16) | (x << 16));
}

void fftSpecial(std::complex<double>* vals, const long size) {

	// int N = CUDAManager::N;
	// int Nh = (N>>1);
	// int M = (N<<1);
	int Nh = CUDAManager::N;
	int N = (Nh << 1);
	int M = (N << 1);

	long *rotGroup = new long[Nh]();
	// #pragma omp parallel for
	for (long i = 0; i < Nh; ++i)
		rotGroup[i] = conv<long>(NTL::PowerMod(to_ZZ(5), i, to_ZZ(M)));

	complex<double> *ksiPows = new complex<double>[M + 1];
	for (long j = 0; j < M; ++j) {
		double angle = 2.0 * M_PI * j / M;
		ksiPows[j].real(cos(angle));
		ksiPows[j].imag(sin(angle));
	}
	ksiPows[M] = ksiPows[0];

	arrayBitReverse(vals, size);
	for (long len = 2; len <= size; len <<= 1) {
		for (long i = 0; i < size; i += len) {
			long lenh = len >> 1;
			long lenq = len << 2;
			for (long j = 0; j < lenh; ++j) {
				long idx = ((rotGroup[j] % lenq)) * M / lenq;
				complex<double> u = vals[i + j];
				complex<double> v = vals[i + j + lenh];
				v *= ksiPows[idx];
				vals[i + j] = u + v;
				vals[i + j + lenh] = u - v;
			}
		}
	}

	delete[] rotGroup;
	delete[] ksiPows;
}

void fftSpecialInvLazy(std::complex<double>* vals, const long size) {

	int N = CUDAManager::N;
	int Nh = (N>>1);
	int M = (N<<1);

	long *rotGroup = new long[Nh]();
	// #pragma omp parallel for
	for (long i = 0; i < Nh; ++i)
		rotGroup[i] = NTL::conv<long>(NTL::PowerMod(NTL::to_ZZ(5), i, NTL::to_ZZ(M)));


	std::complex<double> *ksiPows = new std::complex<double>[M + 1];
	// #pragma omp parallel for
	for (long j = 0; j < M; ++j) {
		double angle = 2.0 * M_PI * j / M;
		ksiPows[j].real(cos(angle));
		ksiPows[j].imag(sin(angle));
	}
	ksiPows[M] = ksiPows[0];

	for (long len = size; len >= 1; len >>= 1)
		for (long i = 0; i < size; i += len) {
			long lenh = len >> 1;
			long lenq = len << 2;
			for (long j = 0; j < lenh; ++j) {
				long idx = (lenq - (rotGroup[j] % lenq)) * M / lenq;

				std::complex<double> u = vals[i + j] + vals[i + j + lenh];
				std::complex<double> v = vals[i + j] - vals[i + j + lenh];
				
				v *= ksiPows[idx];
				
				vals[i + j] = u;
				vals[i + j + lenh] = v;
			}
		}
	arrayBitReverse(vals, size);

	delete[] rotGroup;
	delete[] ksiPows;
}

void fftSpecialInv(std::complex<double>* vals, const long size) {
	fftSpecialInvLazy(vals, size);

	// #pragma omp parallel for
	for (long i = 0; i < size; ++i)
		vals[i] /= size;
}

/////////////
// Encoder //
/////////////
///
#define ADJUSTNEG(x, p) (x >= 0 ?  x % p: p - ((-x) % p))
#define ADJUSTSCALE(x, p, factor) x <= (p>>1) ? ((double) x) / factor : (((double) x) - ((double) p)) / (double) factor

/**
 * Receives an array of complex elements and converts to an array of uint64_t
 *
 * @param b             [description]
 * @param a             [description]
 * @param slots         [description]
 * @param l             [description]
 * @param N             [description]
 * @param scalingfactor [description]
 */
__global__ void ckks_convert_C2PC(
	uint64_t *b,
    cuDoubleComplex *a,
	int slots,	int levels,	int N,
	uint64_t scalingfactor){
	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < slots * levels){
		const int sid = tid % slots; // slot id
		const int lid = tid / slots; // level id
		cuDoubleComplex uval = a[sid];

		int64_t mir = uval.x * scalingfactor;
		int64_t mii = uval.y * scalingfactor;
		int gap = (N>>1) / slots;
		uint64_t p = d_RNSCoprimes[lid];

		b[sid * gap + lid * N] 		  = ADJUSTNEG(mir, p);
		b[sid * gap + (N>>1) + lid * N] = ADJUSTNEG(mii, p);
	}
}

__host__ void ckks_encode(
	CKKSContext *ctx,
	uint64_t* d_output,
	std::complex<double>* h_input,
	int filled_slots,
	int empty_slots,
	uint64_t scalingfactor){

	int slots = filled_slots + empty_slots;
	assert(slots <= (CUDAManager::N>>1));
	assert(is_power2(slots));
	assert(scalingfactor > 0);

    // IFFT
    // # 1 dimensional FFT	
    memcpy(ctx->h_val, 		   h_input, filled_slots * sizeof(std::complex<double>));
    memset(ctx->h_val + filled_slots, 0, empty_slots * sizeof(std::complex<double>));
	fftSpecialInv(ctx->h_val, slots);

	// std::cout << "Special encoded to: " << std::endl;
	// for(int i = 0; i < slots; i++)
	// 	std::cout << ctx->h_val[i] << ", ";
	// std::cout << std::endl;

	cudaMemcpyAsync(
		ctx->d_val,
		ctx->h_val,
		slots * sizeof(std::complex<double>),
		cudaMemcpyHostToDevice,
		ctx->get_stream());
	cudaCheckError();

    // Convert to uint64_t
    const int size = slots * CUDAManager::get_n_residues(QBase);
	const dim3 gridDim(get_grid_dim(size,DEFAULTBLOCKSIZE));
	const dim3 blockDim(DEFAULTBLOCKSIZE);

	cudaMemsetAsync(
		d_output,
		0,
		poly_get_size(QBase),
		ctx->get_stream());
	cudaCheckError();
 	ckks_convert_C2PC<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		d_output,
		(cuDoubleComplex*)ctx->d_val,
		slots,
		CUDAManager::get_n_residues(QBase),
		CUDAManager::N,
		scalingfactor);
	cudaCheckError();

	cudaMemsetAsync(ctx->d_val, 0, slots * sizeof(std::complex<double>), ctx->get_stream());
	cudaCheckError();
	
}


/**
 * Receives an array of uint64_t elements and converts to an array of complex
 *
 * @param b             [description]
 * @param a             [description]
 * @param slots         [description]
 * @param l             [description]
 * @param N             [description]
 * @param scalingfactor [description]
 */
__global__ void ckks_convert_PC2C(
    cuDoubleComplex *b,
	uint64_t *a,
	int slots,
	int N,
	uint64_t scalingfactor){
	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int rid = tid / N;

	if(tid < slots){

		const uint64_t p = d_RNSCoprimes[rid];

		const int sid = tid;
		const int gap = (N>>1) / slots;
		const uint64_t uvalre = a[sid * gap];
		const uint64_t uvalim = a[sid * gap + (N>>1)];

		b[sid] = { 
			ADJUSTSCALE(uvalre, p, scalingfactor),
			ADJUSTSCALE(uvalim, p, scalingfactor)};
	}
}

__host__ void ckks_decode(
	CKKSContext *ctx,
	std::complex<double> *h_val,
	uint64_t *a,
	int slots,
	uint64_t scalingfactor){

	assert(is_power2(slots));
    // Convert to uint64_t
    const int size = slots;
	const dim3 gridDim(get_grid_dim(size,DEFAULTBLOCKSIZE));
	const dim3 blockDim(DEFAULTBLOCKSIZE);

 	ckks_convert_PC2C<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		(cuDoubleComplex*) ctx->d_val,
		a,
		slots, CUDAManager::N,
		scalingfactor);
	cudaCheckError();

	cudaMemcpyAsync(
		h_val,
		ctx->d_val,
		slots * sizeof(std::complex<double>),
		cudaMemcpyDeviceToHost,
		ctx->get_stream());
	cudaCheckError();
	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	// std::cout << "Decoded to: " << std::endl;
	// for(int i = 0; i < slots; i++)
	// 	std::cout << h_val[i] << ", ";
	// std::cout << std::endl;
	
	fftSpecial(h_val, slots);

	cudaMemsetAsync(ctx->d_val, 0, slots * sizeof(std::complex<double>), ctx->get_stream());
	cudaCheckError();
}

__global__ void execute_rotation(
	uint64_t *b,
	uint64_t *a,
	long pow,
	int nresidues,
	int N){
	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int rid = tid / N;
	const int cid = tid % N;

	if (tid < N * nresidues){
		long new_cid = cid * pow % (N<<1);

		int flag = (new_cid >= N); // Is the new cid is out of the bounds?
		new_cid -= SEL(0, N, flag);

		b[new_cid + rid * N] = SEL(
			a[cid + rid * N],
			submod(d_RNSCoprimes[rid], a[cid + rid * N], rid),
			flag);
	}
}


__host__ void rotate_slots_left(
	Context *ctx,
	poly_t *b, // Output
	poly_t *a, // Input
	int rotSlots){

	int N = CUDAManager::N;
	int Nh = (N>>1);
	int M = (N<<1);
	int nresidues = CUDAManager::get_n_residues(a->base);
	
	// Pre-compute the map
	long rotGroup = NTL::conv<long>(NTL::PowerMod(NTL::to_ZZ(5), rotSlots, NTL::to_ZZ(M)));

	// Remove this layer
	COMMONEngine::execute(ctx, a, INVERSE);

	// Call CUDA kernel
    const int size = N * nresidues;
	const dim3 gridDim(get_grid_dim(size,DEFAULTBLOCKSIZE));
	const dim3 blockDim(DEFAULTBLOCKSIZE);

	execute_rotation<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		b->d_coefs, 
		a->d_coefs,
		rotGroup,
		nresidues, N);
  	cudaCheckError();
}


__host__ void rotate_slots_right(
	Context *ctx,
	poly_t *b,
	poly_t *a,
	int rotSlots){

	rotate_slots_left(ctx, b, a, (CUDAManager::N>>1) - rotSlots);
}


// __global__ void execute_conjugation(
// 	uint64_t *b,
// 	uint64_t *a,
// 	int N,
// 	int nresidues){
	
// 	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
// 	const int Nh = (N>>1);
// 	const int rid = tid / Nh;
// 	const int cid = tid % Nh;

// 	if (tid < Nh * nresidues){
// 		int cid_re = (Nh - cid) % Nh;
// 		int cid_im = (Nh - cid) % Nh + Nh;

// 		if(cid == 0){
// 			b[cid + 	 rid * N] = a[cid_re + rid * N];
// 			b[cid + Nh + rid * N] = submod(d_RNSCoprimes[rid], a[cid_im + rid * N], rid);
// 		}else{
// 			b[cid + 	 rid * N] = submod(d_RNSCoprimes[rid], a[cid_im + rid * N], rid);
// 			b[cid + Nh + rid * N] = submod(d_RNSCoprimes[rid], a[cid_re + rid * N], rid);			
// 		}
// 	}
// }

__global__ void execute_conjugation(
	uint64_t *b, uint64_t *a,
	int N, int nresidues){
	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int rid = tid / N;
	const int cid = tid % N;

	if (tid < N * nresidues)
		b[cid + rid * N] = a[(N - 1 - cid) + rid * N];
}


__host__ void conjugate_slots(
	Context *ctx,
	poly_t *a){

	// Remove this layer
	COMMONEngine::execute(ctx, a, FORWARD);

	// Call CUDA kernel
    const int size = CUDAManager::N * CUDAManager::get_n_residues(a->base); 
	const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
	const dim3 blockDim(DEFAULTBLOCKSIZE);
	
	execute_conjugation<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		ctx->d_tmp, a->d_coefs,
		CUDAManager::N, CUDAManager::get_n_residues(a->base));
  	cudaCheckError();

	cudaMemcpyAsync(
		a->d_coefs,	ctx->d_tmp,
		poly_get_size(a->base),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream());
  	cudaCheckError();
}