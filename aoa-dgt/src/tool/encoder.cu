#include <AOADGT/tool/encoder.h>
#include <AOADGT/settings.h>

extern __constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];

bool is_power_of_2(int n){
	return (n>0 && ((n & (n-1)) == 0));
}

void arrayBitReverse(complex<double>* vals, const long size) {
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

void fftSpecial(complex<double>* vals, const long size) {

	int Nh = CUDAEngine::N;
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

void fftSpecialInvLazy(complex<double>* vals, const long size) {

	int Nh = CUDAEngine::N;
	int N = (Nh << 1);
	int M = (N << 1);

	long *rotGroup = new long[Nh]();
	// #pragma omp parallel for
	for (long i = 0; i < Nh; ++i)
		rotGroup[i] = conv<long>(NTL::PowerMod(to_ZZ(5), i, to_ZZ(M)));


	complex<double> *ksiPows = new complex<double>[M + 1];
	// #pragma omp parallel for
	for (long j = 0; j < M; ++j) {
		double angle = 2.0 * M_PI * j / M;
		ksiPows[j].real(cos(angle));
		ksiPows[j].imag(sin(angle));
	}

	for (long len = size; len >= 1; len >>= 1) {
		for (long i = 0; i < size; i += len) {
			long lenh = len >> 1;
			long lenq = len << 2;
			for (long j = 0; j < lenh; ++j) {
				long idx = (lenq - (rotGroup[j] % lenq)) * M / lenq;
				complex<double> u = vals[i + j] + vals[i + j + lenh];
				complex<double> v = vals[i + j] - vals[i + j + lenh];
				v *= ksiPows[idx];
				vals[i + j] = u;
				vals[i + j + lenh] = v;
			}
		}
	}
	arrayBitReverse(vals, size);

	delete[] rotGroup;
	delete[] ksiPows;
}

void fftSpecialInv(complex<double>* vals, const long size) {
	fftSpecialInvLazy(vals, size);

	// #pragma omp parallel for
	for (long i = 0; i < size; ++i)
		vals[i] /= size;
}

__host__ void ckks_encode_single(
	GaussianInteger *a,
	std::complex<double> val,
	uint64_t encodingp){

	uint64_t q = encodingp;
	int64_t vr = real(val) * q;
	int64_t vi = imag(val) * q;
	int L = CUDAEngine::RNSPrimes.size() - 1;

	for (long i = 0; i <= L; ++i) {
		a[i * CUDAEngine::N].re = (
			vr >= 0 ? vr % CUDAEngine::RNSPrimes[i]: CUDAEngine::RNSPrimes[i] - ((-vr) % CUDAEngine::RNSPrimes[i])
			);
		a[i * CUDAEngine::N].imag = (
			vi >= 0 ? vi % CUDAEngine::RNSPrimes[i]: CUDAEngine::RNSPrimes[i] - ((-vi) % CUDAEngine::RNSPrimes[i])
			);
	}	
}

__host__ void ckks_decode_single(
	std::complex<double> *val,
	GaussianInteger *a,
	uint64_t encodingp){
	uint64_t q =  encodingp;
	uint64_t pr = CUDAEngine::RNSPrimes[0];
	uint64_t pr_2 = CUDAEngine::RNSPrimes[0] / 2;

	double vr = a[0].re <= pr_2 ?
		((double) a[0].re) / q :
		(((double) a[0].re) - ((double) pr)) / (double) q;
	double vi = a[0].imag <= pr_2 ?
		((double) a[0].imag) / q :
		(((double) a[0].imag) - ((double) pr)) / (double) q;

	val->real(vr);
	val->imag(vi);
}

__global__ void ckks_convert_C2GI(
	GaussianInteger *b,
    cuDoubleComplex *a,
	int slots,
	int l,
	int N,
	uint64_t encodingp,
	uint64_t *RNSCoprimes){
	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid <= slots * l){
		const int sid = tid % slots; // slot id
		const int lid = tid / slots; // level id
		cuDoubleComplex uval = a[sid];

		int64_t mir = uval.x * encodingp;
		int64_t mii = uval.y * encodingp;
		int gap = N / slots;

		b[sid * gap + lid * N] = {
			mir >= 0 ?
				(uint64_t) mir % RNSCoprimes[lid] : RNSCoprimes[lid] - (uint64_t) ((-mir) % RNSCoprimes[lid]),
			mii >= 0 ?
				(uint64_t) mii % RNSCoprimes[lid]: RNSCoprimes[lid] - (uint64_t) ((-mii) % RNSCoprimes[lid])
			};
	}
}

__host__ void ckks_encode(
	CKKSContext *ctx,
	GaussianInteger* d_output,
	std::complex<double>* h_input,
	int filled_slots,
	int empty_slots,
	uint64_t encodingp){

	int slots = filled_slots + empty_slots;
	assert(slots <= CUDAEngine::N);
	assert(is_power_of_2(slots));

    // IFFT
    // # 1 dimensional FFT	
    memcpy(ctx->h_val, 		   h_input, filled_slots * sizeof(complex<double>));
    memset(ctx->h_val + filled_slots, 0, empty_slots * sizeof(complex<double>));
	fftSpecialInv(ctx->h_val, slots);

	// std::cout << "Special encoded to: " << std::endl;
	// for(int i = 0; i < slots; i++)
	// 	std::cout << ctx->h_val[i] << ", ";
	// std::cout << std::endl;

 //    memcpy(ctx->h_val, 		   h_input, filled_slots * sizeof(complex<double>));
 //    memset(ctx->h_val + filled_slots, 0, empty_slots * sizeof(complex<double>));

 //    for(int i = 0; i < filled_slots; i++)
 //    	ctx->h_val[i] = conj(ctx->h_val[i]);
	// fftSpecialInv(ctx->h_val, slots);

	// std::cout << "Conjugate special encoded to: " << std::endl;
	// for(int i = 0; i < slots; i++)
	// 	std::cout << ctx->h_val[i] << ", ";
	// std::cout << std::endl;
 	
	// fftw_plan plan = fftw_plan_dft_1d(
	// 	slots,
	// 	(fftw_complex*)h_input,
	// 	(fftw_complex*)ctx->h_val,
	// 	FFTW_BACKWARD,
	// 	FFTW_ESTIMATE);
	// fftw_execute(plan);
	// fftw_destroy_plan(plan);

	// std::cout << "FFTW encoded to: " << std::endl;
	// for(int i = 0; i < slots; i++)
	// 	std::cout << ctx->h_val[i] << ", ";
	// std::cout << std::endl;

	cudaMemcpyAsync(
		ctx->d_val,
		ctx->h_val,
		slots * sizeof(complex<double>),
		cudaMemcpyHostToDevice,
		ctx->get_stream());
	cudaCheckError();

    // Convert to GaussianInteger
    const int size = slots * CUDAEngine::get_n_residues(QBase);
  	const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	cudaMemsetAsync(
		d_output,
		0,
		poly_get_size(QBase),
		ctx->get_stream());
	cudaCheckError();
 	ckks_convert_C2GI<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		d_output,
		(cuDoubleComplex*)ctx->d_val,
		slots,
		CUDAEngine::get_n_residues(QBase),
		CUDAEngine::N,
		encodingp,
		CUDAEngine::RNSCoprimes);
	cudaCheckError();
	
}

__global__ void ckks_convert_GI2C(
    cuDoubleComplex *b,
	GaussianInteger *a,
	int slots,
	int N,
	uint64_t encodingp,
	uint64_t *RNSCoprimes){

	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < slots){

		const int sid = tid % slots;
		const int rid = tid / N;
		int gap = N / slots;
		GaussianInteger uval = a[sid * gap];

		b[sid] = {
			uval.re <= RNSCoprimes[rid]/2 ?
				(double) uval.re / encodingp :
				((double) uval.re - (int64_t)(RNSCoprimes[rid])) / encodingp,
			uval.imag <= RNSCoprimes[rid]/2 ?
				(double) uval.imag / encodingp :
				((double) uval.imag - (int64_t)(RNSCoprimes[rid])) / encodingp
		};
	}
}

__host__ void ckks_decode(
	CKKSContext *ctx,
	std::complex<double> *h_val,
	GaussianInteger *a,
	int slots,
	uint64_t encodingp){

	assert(is_power_of_2(slots));
    // Convert to GaussianInteger
    const int size = slots;
  	const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);
 	ckks_convert_GI2C<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		(cuDoubleComplex*)ctx->d_val,
		a,
		slots,
		CUDAEngine::N,
		encodingp,
		CUDAEngine::RNSCoprimes);
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

	// fftw_plan plan = fftw_plan_dft_1d(
	// 	slots,
	// 	(fftw_complex*)h_val,
	// 	(fftw_complex*)h_val,
	// 	FFTW_FORWARD,
	// 	FFTW_ESTIMATE);
	// fftw_execute(plan);

}


__global__ void execute_rotation(
	GaussianInteger *b,
	GaussianInteger *a,
	long pow,
	int nresidues,
	int N,
	uint64_t *RNSCoprimes){
	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int rid = tid / N;
	const int cid = tid % N;

	if (tid < N * nresidues){
		long new_cid_re = cid 		* pow % (N<<2);
		long new_cid_im = (cid + N) * pow % (N<<2);
		int flag_re = (new_cid_re >= (N<<1)); // Is the new cid is out of the bounds?
		int flag_im = (new_cid_im >= (N<<1)); // Is the new cid is out of the bounds?
		new_cid_re -= SEL(0, (N<<1), flag_re);
		new_cid_im -= SEL(0, (N<<1), flag_im);

		GaussianInteger coef = a[cid + rid * N];

		b[(new_cid_re%N) + rid * N].write(new_cid_re >= N, SEL(
			coef.re,
			SEL(coef.re, RNSCoprimes[rid] - coef.re, coef.re < RNSCoprimes[rid]),
			flag_re));

		b[(new_cid_im%N) + rid * N].write(new_cid_im >= N, SEL(
			coef.imag,
			SEL(coef.imag, RNSCoprimes[rid] - coef.imag, coef.imag < RNSCoprimes[rid]),
			flag_im));
	}
}

__host__ void rotate_slots_right(
	CKKSContext *ctx,
	poly_t *b, // Output
	poly_t *a, // Input
	int rotSlots){

	rotate_slots_left(ctx, b, a, CUDAEngine::N - rotSlots);
}


__host__ void rotate_slots_left(
	CKKSContext *ctx,
	poly_t *b, // Output
	poly_t *a, // Input
	int rotSlots){

	int Nh = CUDAEngine::N;
	int N = (Nh << 1);
	int M = (N << 1);
	int nresidues = CUDAEngine::get_n_residues(a->base);
	
	// Pre-compute the map
	uint64_t rotGroup = conv<long>(NTL::PowerMod(to_ZZ(5), rotSlots, to_ZZ(M)));

	// Remove the DGT layer
	DGTEngine::execute_dgt( ctx, a, INVERSE);

	// Call CUDA kernel
    const int size = Nh * nresidues;
  	const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	execute_rotation<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		b->d_coefs,
		a->d_coefs,
		rotGroup,
		nresidues,
		Nh,
		CUDAEngine::RNSCoprimes);
  	cudaCheckError();

  	b->base = a->base;
  	b->state = a->state;	
}


__global__ void execute_conjugation(
	GaussianInteger *b,
	GaussianInteger *a,
	int N,
	int nresidues,
	uint64_t *RNSCoprimes){
	
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int rid = tid / N;
	const int cid = tid % N;
	int cidx = (N - cid) % N;

	if (tid < N * nresidues){
		GaussianInteger x = a[cidx + rid * N];

		if(cid == 0)
			b[cid + rid * N] = (GaussianInteger){
				x.re, RNSCoprimes[rid] - x.imag
			};
		else
			b[cid + rid * N] = (GaussianInteger){
				RNSCoprimes[rid] - x.imag, RNSCoprimes[rid] - x.re
			};
	}
}


__host__ void conjugate_slots(
	CKKSContext *ctx,
	poly_t *a){

	// Remove the DGT layer
	DGTEngine::execute_dgt( ctx, a, INVERSE );

	// Call CUDA kernel
	const int NResidues = CUDAEngine::get_n_residues(a->base);
    const int size = CUDAEngine::N * NResidues; 
  	const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	cudaMemcpyAsync(ctx->d_aux_coefs, a->d_coefs, size * sizeof(GaussianInteger), cudaMemcpyDeviceToDevice, ctx->get_stream());
  	cudaCheckError();

	execute_conjugation<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
		a->d_coefs,
		ctx->d_aux_coefs,
		CUDAEngine::N,
		NResidues,
		CUDAEngine::RNSCoprimes);
  	cudaCheckError();

}