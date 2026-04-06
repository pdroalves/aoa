#include <AOADGT/cuda/dgt.h>
#include <AOADGT/arithmetic/context.h>

bool DGTEngine::is_init;
std::map<int, uint64_t*> DGTEngine::d_gjk;
std::map<int, uint64_t*> DGTEngine::d_invgjk;
std::map<int, GaussianInteger*> DGTEngine::d_nthroot;
std::map<int, GaussianInteger*> DGTEngine::d_invnthroot;
std::map<int, GaussianInteger*> DGTEngine::h_nthroot;
std::map<int, GaussianInteger*> DGTEngine::h_invnthroot;
uint64_t *DGTEngine::d_gN = NULL;
uint64_t *DGTEngine::d_ginvN = NULL;

__constant__ int d_RNSCoprimes_NumBits[MAX_COPRIMES];
int RNSCoprimes_NumBits[MAX_COPRIMES];
__constant__ uint64_t d_BARRETT_MU[MAX_COPRIMES];
uint64_t h_BARRETT_MU[MAX_COPRIMES]; // Debug purposes only

NTL_CLIENT

#ifndef max

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )

#endif

// Returns the position of the highest bit
__host__ int hob (int num)
{
    if (!num)
        return 0;

    int ret = 1;

    while (num >>= 1)
        ret++;

    return ret-1;
}                

void print_residue(
  GaussianInteger *d_data,
  int N,
  int rid,
  cudaStream_t stream){

  cudaStreamSynchronize(stream);
  cudaCheckError();

  GaussianInteger *h_data = (GaussianInteger*) malloc (N * sizeof(GaussianInteger));
  cudaMemcpy(h_data, d_data + rid * CUDAEngine::N * sizeof(GaussianInteger), N * sizeof(GaussianInteger), cudaMemcpyDeviceToHost);
  cudaCheckError();

  std::cout << "[ ";
  for(int i = 0; i < N; i++)
      std::cout << h_data[i].re << ", ";
  std::cout << "]" << std::endl;

  free(h_data);
}

void print_residue_matrix(
  GaussianInteger *d_data,
  int Na,
  int Nb,
  int rid,
  cudaStream_t stream,
  bool transp = false,
  int matrix_size = 8){
  assert(Na * Nb == CUDAEngine::N);
  cudaStreamSynchronize(stream);
  cudaCheckError();

  GaussianInteger *h_data = (GaussianInteger*) malloc (CUDAEngine::N * sizeof(GaussianInteger));
  cudaMemcpy(h_data, d_data + rid * CUDAEngine::N, CUDAEngine::N * sizeof(GaussianInteger), cudaMemcpyDeviceToHost);
  cudaCheckError();

  for(int i = 0; i < matrix_size; i++){ // Row
    for(int j = 0; j < matrix_size; j++) // Column
      if(transp)
        std::cout << (h_data[i + j * Nb].re % COPRIMES_BUCKET[rid]) << ", ";
      else
        std::cout << (h_data[j + i * Nb].re % COPRIMES_BUCKET[rid]) << ", ";
    std::cout  << std::endl;
  }

  free(h_data);
}

// These macros improves readability at execute_dgt
#define runDGTKernel(N)                                   \
  if(direction == FORWARD)                                \
    dgt<N><<< nresidues, (N >> 1), 0, ctx->get_stream()>>>(    \
      data,                                               \
      nresidues,                                          \
      rid_offset,                                         \
      hob(N),                                             \
      DGTEngine::d_gjk[N],                                \
      DGTEngine::d_nthroot[N]);                           \
  else                                                    \
    idgt<N><<< nresidues, (N >> 1), 0, ctx->get_stream()>>>(\
      data,                                               \
      nresidues,                                          \
      rid_offset,                                         \
      hob(N),                                             \
      DGTEngine::d_invgjk[N],                             \
      DGTEngine::d_invnthroot[N]);       

#define runDGTKernelHierarchical(Na, Nb)                                        \
  if(direction == FORWARD){                                                     \
    hdgt_tr<Nb><<< Na * nresidues, Nb/2, 0, ctx->get_stream()>>>(               \
      ctx->d_tmp_data,                                                    \
      data,                                                                     \
      nresidues,                                                                \
      rid_offset,                                                               \
      hob(Nb),                                                                  \
      DGTEngine::d_nthroot[Na * Nb] + CUDAEngine::N * rid_offset,               \
      DGTEngine::d_gjk[Nb]);                                                    \
    hdgt_tr<Na><<< Nb * nresidues, Na/2, 0, ctx->get_stream()>>>(               \
      data,                                                                     \
      ctx->d_tmp_data,                                                    \
      nresidues,                                                                \
      rid_offset,                                                               \
      hob(Na),                                                                  \
      DGTEngine::d_gN + CUDAEngine::N * rid_offset,                             \
      DGTEngine::d_gjk[Na]);                                                    \
  }else{                                                                        \
    hidgt<Na><<< Nb * nresidues, Na/2, 0, ctx->get_stream()>>>(                 \
      ctx->d_tmp_data,                                                    \
      data,                                                                     \
      nresidues,                                                                \
      rid_offset,                                                               \
      hob(Na),                                                                  \
      DGTEngine::d_ginvN + CUDAEngine::N * rid_offset,                          \
      DGTEngine::d_invgjk[Na]);                                                 \
    hidgt_tr<Nb><<< Na * nresidues, Nb/2, 0, ctx->get_stream()>>>(              \
      data,                                                                     \
      ctx->d_tmp_data,                                                    \
      nresidues,                                                                \
      rid_offset,                                                               \
      hob(Nb),                                                                  \
      DGTEngine::d_invnthroot[Na * Nb] + CUDAEngine::N * rid_offset,            \
      DGTEngine::d_invgjk[Nb]);                                                 \
    }       

#define runDGT(N)   \
  switch(N){        \
    case 8:        \
    runDGTKernel(8);\
    break;          \
    case 16:        \
    runDGTKernel(16);\
    break;          \
    case 32:        \
    runDGTKernel(32);  \
    break;          \
    case 64:        \
    runDGTKernel(64);  \
    break;          \
    case 128:       \
    runDGTKernel(128); \
    break;          \
    case 256:       \
    runDGTKernel(256); \
    break;          \
    case 512:       \
    runDGTKernel(512); \
    break;          \
    case 1024:      \
    if(ENABLE_HDGT_1024) runDGTKernelHierarchical(DGTFACT1024, DGTFACT1024) else runDGTKernel(1024); \
    break;          \
    case 2048:      \
    if(ENABLE_HDGT_2048) runDGTKernelHierarchical(DGTFACT2048a, DGTFACT2048b) else runDGTKernel(2048); \
    break;          \
    case 4096:      \
    runDGTKernelHierarchical(DGTFACT4096, DGTFACT4096);\
    break;          \
    case 8192:      \
    runDGTKernelHierarchical(DGTFACT8192a, DGTFACT8192b);\
    break;          \
    case 16384:      \
    runDGTKernelHierarchical(DGTFACT16384, DGTFACT16384);\
    break;          \
    case 32768:      \
    runDGTKernelHierarchical(DGTFACT32768a, DGTFACT32768b);\
    break;          \
    default:        \
    throw std::runtime_error("DGT is not working for such N");\
  }
////////////////////////////////////////////
// Basic operators for modular arithmetic //
////////////////////////////////////////////

// Comparison
__host__ __device__ bool operator==(const uint128_t &a, const uint128_t &b){
  if(a.hi == b.hi && a.lo == b.lo)
    return true;
  else
    return false;
}

__host__ __device__ bool operator!=(const uint128_t& a, const uint128_t& b){
  return !(a == b);
}

__host__ __device__ bool operator<(const uint128_t &a, const uint128_t &b){
    if(a.hi < b.hi)
      return true;
    else if(a.hi > b.hi)
      return false;
    else{
      if(a.lo < b.lo)
        return true;
      else
        return false;
    }
}


__host__ __device__ bool operator>(const uint128_t &a, const uint128_t &b){
  if(a != b && !(a < b))
    return true;
  else
    return false;
}

__host__ __device__ bool operator<=(const uint128_t &a, const uint128_t &b){
  if(a < b || a == b)
    return true;
  else
    return false;
}

__host__ __device__ bool operator>=(const uint128_t &a, const uint128_t &b){
  if(a > b || a == b)
    return true;
  else
    return false;
}

__host__ __device__ bool operator>=(const uint128_t &a, const uint64_t &b){
  if(a.hi != 0)
    return true;
  else
    return (a.lo >= b);
}

// Arithmetic
__host__ __device__ uint128_t operator-(const uint128_t &a, const uint128_t &b){
  uint128_t c;

  c.lo = a.lo - b.lo;
  c.hi = (a.hi - b.hi) - (a.lo < b.lo);

  return c;
}


__host__ __device__ inline void load_coprime(uint64_t *p, int *numbits, int rid){
#ifdef __CUDA_ARCH__
  *p = d_RNSCoprimes[rid];
  *numbits = d_RNSCoprimes_NumBits[rid];
#else
  *p = COPRIMES_BUCKET[rid];
  *numbits = RNSCoprimes_NumBits[rid];
#endif
 }

__host__ __device__ inline uint64_t mul64hi(uint64_t a, uint64_t b) {
  #ifdef __CUDA_ARCH__
  return __umul64hi(a, b);
  #else
  unsigned __int128 prod =  a * (unsigned __int128)b;
  return prod >> 64;
  #endif
 }

/**
 * @brief      Multiply a and b
 *
 * @param[in]  a     { parameter_description }
 * @param[in]  b     { parameter_description }
 *
 * @return     { description_of_the_return_value }
 */
__host__ __device__  uint128_t mullazy(const uint64_t a, const uint64_t b){
  uint128_t c;
  c.hi = mul64hi(a,b);
  c.lo = a * b;

  return c;
}

__host__ __device__ inline uint128_t add128(uint128_t x, uint128_t y, int rid){
  // uint64_t p;
  // int numbits;
  // load_coprime(&p, &numbits, rid);
  // uint128_t P;

  uint128_t z;
  z.lo = (x.lo + y.lo);  
  z.hi = (x.hi + y.hi) + (z.lo < x.lo); 

  // int overflow = (z.hi < x.hi || z.hi < y.hi);
  // P.lo = (p<<numbits) * overflow;
  // P.hi = (p>>(64-numbits)) * overflow;
  
  // uint128_t res;
  // res.lo = z.lo - P.lo;
  // res.hi = z.hi - P.hi - (z.lo < P.lo);
  return z;
}

// sub128 partially extracted from https://github.com/curtisseizert/CUDA-uint128
__host__ __device__ uint128_t sub128(uint128_t x, uint128_t y, int rid) // x - y
  {
    uint64_t p;
    int numbits;
    load_coprime(&p, &numbits, rid);
    uint128_t P;

    // Subtraction
    uint128_t z;
    z.lo = x.lo - y.lo;
    z.hi = x.hi - y.hi - (x.lo < y.lo);

    // Adds (p<<NumBits(p)) to fix overflows
    int underflow = (x.hi < y.hi || (x.hi == y.hi && x.lo < y.lo));
    P.lo = (p<<numbits) * underflow;
    P.hi = (p>>(64-numbits)) * underflow;

    uint128_t res;
    res.lo = (z.lo + P.lo);  
    res.hi = (z.hi + P.hi) + (res.lo < z.lo); 
    return res;
  }

/**
 * @brief      left shift
 *
 * @param      x     { parameter_description }
 * @param[in]  b     { parameter_description }
 */
__host__ __device__ inline uint128_t lshift(uint128_t x, int b){
  uint128_t y = x; 
  
  y.hi = (y.hi << b) + (y.lo >> (WORDSIZE - b));
  y.lo = (y.lo << b);

  return y;
}

/**
 * @brief      right shift
 *
 * @param[in]  x     { parameter_description }
 * @param[in]  b     { parameter_description }
 *
 * @return     { description_of_the_return_value }
 */
__host__ __device__ inline uint128_t rshift(uint128_t x, int b){
  uint128_t y = x; 
  
  y.lo = (y.lo >> b) + (y.hi << (WORDSIZE - b));
  y.hi = (y.hi >> b);

  return y;
}

/**
 * @brief      Computes a mod p, for the rid-th coprime
 *
 * @param[in]  a     128 bits uint64_t
 * @param[in]  rid   coprime index
 *
 * @return     { description_of_the_return_value }
 */
__host__ __device__ uint64_t mod(uint128_t a, int rid){

  // Load the related coprime
  uint64_t p; int numbits;
  load_coprime(&p, &numbits, rid);

  uint64_t z = ((uint64_t)1<<numbits) - 1;
  // Adjust to numbits bits words
  uint64_t m0 = a.lo & z;
  uint64_t m1 = (a.hi << (64-numbits)) + (a.lo >> numbits) ;

  // Mod
  // q has the numbits msb bits of m1 * MU
  // q0 has the numbits lsb bits of q*p
  // q1 has the numbits msb of q*p
  #ifdef __CUDA_ARCH__
  uint64_t MU = d_BARRETT_MU[rid];
  #else
  uint64_t MU = h_BARRETT_MU[rid];
  #endif

  uint128_t x = rshift(a, 2 * numbits - 64);
  uint64_t q  = (mul64hi(x.lo, MU) + (x.hi * MU));
  uint128_t qp = mullazy(q, p);

  uint64_t q0 = qp.lo & z;
  uint64_t q1 = (qp.hi << (64 - numbits)) + (qp.lo >> numbits);

  uint64_t r0 = (m0 - q0);
  uint64_t r1 = (m1 - q1) - (m0 < q0);
  uint64_t r  = (r0 - r1*p) & z;

  return SEL(r, r-p, (r>=p));

}


/**
 * @brief      Multiply and reduce a and b by p
 *
 * @param[in]  a     { parameter_description }
 * @param[in]  b     { parameter_description }
 * @param[in]  rid   The rid
 *
 * @return     { description_of_the_return_value }
 */
__host__ __device__  uint64_t mulmod(const uint64_t a, const uint64_t b, const int rid){
  uint128_t c;
  c.hi = mul64hi(a,b);
  c.lo = a * b;

  #ifdef NDEBUG
  #ifdef __CUDA_ARCH__
    uint64_t p = d_RNSCoprimes[rid];
  #else
    uint64_t p = COPRIMES_BUCKET[rid];
  #endif
    
  if(a > p)
    printf("a >= p: %lu >= %lu\n", a, p);
  if(b >= p)
    printf("b >= p: %lu >= %lu\n", b, p);
  #endif
  
  return mod(c, rid);
}

/**
 * @brief      Computes a * b mod p
 *
 * @param      c       The answer
 * @param[in]  a       A Gaussian uint64_t
 * @param[in]  b       A integer
 * @param[in]  rid     An index for a residue
 */
__device__ void mulint_dgt(
  GaussianInteger *c,
  const GaussianInteger a,
  const uint64_t b,
  const int rid){

  *c = (GaussianInteger){
      mulmod(a.re, b, rid),
      mulmod(a.imag, b, rid)
    };
}

/**
 * @brief      Computes a * b mod p
 *
 * @param      c       The answer
 * @param[in]  a       A Gaussian uint64_t
 * @param[in]  b       A integer
 * @param[in]  rid     An index for a residue
 */
__device__ GaussianInteger mulint_dgt(
  const GaussianInteger a,
  const uint64_t b,
  const int rid){

  return (GaussianInteger){
      mulmod(a.re, b, rid),
      mulmod(a.imag, b, rid)
    };
}

// Computes a ^ b mod p
__host__ __device__ uint64_t fast_pow(uint64_t a, uint64_t b, int rid){
  uint64_t r = 1;
  uint64_t s = a;
  while(b > 0){
    if(b % 2 == 1)
      r = mulmod(r, s, rid);
    s = mulmod(s, s, rid);
    b /= 2;
  }
  return r;
}



////////////////////////////////////////////////////////////////////////////////
// Gaussian uint64_t operations
////////////////////////////////////////////////////////////////////////////////

__host__ __device__ GaussianInteger GIAdd(GaussianInteger a, GaussianInteger b, int rid){
    GaussianInteger c;
    GIAdd(&c, a, b, rid);
    return c;
}

__host__ __device__ void GIAdd(GaussianInteger *c, GaussianInteger a, GaussianInteger b, int rid){
    c->re = addmod(a.re, b.re, rid);
    c->imag = addmod(a.imag, b.imag, rid);
}


__host__ __device__ GaussianInteger GISub(GaussianInteger a, GaussianInteger b, int rid){
    GaussianInteger c;
    GISub(&c, a, b, rid);
    return c;
}


__host__ __device__ void GISub(GaussianInteger *c, GaussianInteger a, GaussianInteger b, int rid){
    c->re = submod(a.re, b.re, rid);
    c->imag = submod(a.imag, b.imag, rid);
}

// #define LAZYREDUCTION
#ifndef LAZYREDUCTION
// GaussianInteger multiplication
__host__ __device__ GaussianInteger GIMul(GaussianInteger a, GaussianInteger b, int rid){
  // Karatsuba method
  // https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
  // 
  // S1=ac,S2=bd, and S3=(a+b)(c+d). Now you can compute the results as 
  // A=S1−S2 and B=S3−S1−S2.
  // 
    uint64_t s1 = mulmod(a.re, b.re, rid);
    uint64_t s2 = mulmod(a.imag, b.imag, rid);
    uint64_t s3 = mulmod(
      addmod(a.re, a.imag, rid),
      addmod(b.re, b.imag, rid),
      rid
      );

    uint64_t cre = submod(s1, s2, rid);
    uint64_t cimag = submod(
      s3,
      addmod(s1, s2, rid),
      rid);
    
    // Lazy reduction
    return (GaussianInteger){cre, cimag};
}
#else
// GaussianInteger multiplication
__host__ __device__ GaussianInteger GIMulNotLazy(GaussianInteger a, GaussianInteger b, int rid){
  // Karatsuba method
  // https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
  // 
  // S1=ac,S2=bd, and S3=(a+b)(c+d). Now you can compute the results as 
  // A=S1−S2 and B=S3−S1−S2.
  // 
    uint64_t s1 = mulmod(a.re, b.re, rid);
    uint64_t s2 = mulmod(a.imag, b.imag, rid);
    uint64_t s3 = mulmod(
      addmod(a.re, a.imag, rid),
      addmod(b.re, b.imag, rid),
      rid
      );

    uint64_t cre = submod(s1, s2, rid);
    uint64_t cimag = submod(
      s3,
      addmod(s1, s2, rid),
      rid);
    
    // Lazy reduction
    return (GaussianInteger){cre, cimag};
}
// mod() may return the wrong outcome in this version.
// 
// [ RUN      ] AOADGTInstantiation/TestRNS.SimpleBasisExtensionUniformQtoQB/256_q120
// 4) Errou! 154834404, 339725720 != 31233509, 339725720
// a: 6728798962089412336, 1248370673306852321
// b: 736958053817342579, 7626031173833082297
// s1: 6646667918339816912, 268819395326080902
// s2: 6653126247567985305, 516086396227787052
// s3: 8294461199219137516, 3616518094542164276
// cre: 9216913707626607415, 4364419017463881306
// cimag: 13441411107020886915, 2831612302988296321
// /home/pedro/spogpack/AOADGT/tests/unit_tests/test_bfv_mode.cu:867: Failure
// Expected equality of these values:
//   a_QB_h_coefs[i + (rid + CUDAEngine::RNSPrimes.size())*(CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid]
//     Which is: 31233509
//   poly_get_coeff(ctx, &a_Q, i) % CUDAEngine::RNSBPrimes[rid]
//     Which is: 154834404
// Fail at index 13, rid: 2

__host__ __device__ GaussianInteger GIMul(GaussianInteger a, GaussianInteger b, int rid){
  // Karatsuba method
  // https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
  // 
  // S1=ac,S2=bd, and S3=(a+b)(c+d). Now you can compute the results as 
  // A=S1−S2 and B=S3−S1−S2.
  // 
    uint128_t s1 = mullazy(a.re, b.re);
    uint128_t s2 = mullazy(a.imag, b.imag);
    uint128_t s3 = mullazy(
      addmod(a.re, a.imag, rid),
      addmod(b.re, b.imag, rid)
      );

    uint128_t cre = sub128(s1, s2, rid);
    uint128_t cimag = sub128(
      s3,
      add128(s1, s2, rid),
      rid);

    GaussianInteger res = (GaussianInteger){mod(cre, rid), mod(cimag, rid)};
    GaussianInteger expected = GIMulNotLazy(a,b,rid);
    if(res.re != expected.re || res.imag != expected.imag){
      printf("%d) Errou! %lu, %lu != %lu, %lu\n", rid, expected.re, expected.imag, res.re, res.imag);
      printf("a: %lu, %lu\n", a.re, a.imag);
      printf("b: %lu, %lu\n", b.re, b.imag);
      printf("s1: %lu, %lu\n", s1.lo, s1.hi);
      printf("s2: %lu, %lu\n", s2.lo, s2.hi);
      printf("s3: %lu, %lu\n", s3.lo, s3.hi);
      printf("cre: %lu, %lu\n", cre.lo, cre.hi);
      printf("cimag: %lu, %lu\n", cimag.lo, cimag.hi);
      uint64_t res = mod(sub128(s1, s2, rid),rid);
    }
    
    // Lazy reduction
    return res;
}
#endif

// GaussianInteger multiplication by constant
__host__ __device__ void GIMul(GaussianInteger *c, GaussianInteger a, uint64_t b, int rid){
  c->re = mulmod(a.re, b, rid);
  c->imag = mulmod(a.imag, b, rid);
}

__host__ __device__ GaussianInteger GIMul(GaussianInteger a, uint64_t b, int rid){
  GaussianInteger c;
  GIMul(&c, a, b, rid);
  return c;
}

// Computes a ^ b mod p[rid]
__host__ __device__ GaussianInteger GIFastPow(GaussianInteger a, int b, int rid){
  GaussianInteger r = {1, 0};
  GaussianInteger s = a;
  while(b > 0){
    if(b % 2 == 1)
      r = GIMul(r, s, rid);
    s = GIMul(s, s, rid);
    b /= 2;
  }
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// Discrete Galois transform
////////////////////////////////////////////////////////////////////////////////

__device__ void dgttransform(
  GaussianInteger* data,
  const int cid,
  const int rid,
  const int stride,
  const int N,
  const int nresidues,
  const int m,
  const int log2k,
  const uint64_t *gjk){

  // Indexes
  const int j = cid * 2 * m / N;
  const int i = j + (cid % (N/(2*m)))*2*m;
  const int xi_index = i;
  const int xim_index = i + m;

  // Coefs
  GaussianInteger xi = data[xi_index];
  GaussianInteger xim = data[xim_index];

  // DGT
  const uint64_t a = gjk[(rid * log2k + stride) * N + j];
  
  // Write the result
  data[xi_index]           = GIAdd(xi, xim, rid); 
  GaussianInteger xisubxim = GISub(xi, xim, rid);
  data[xim_index] = (GaussianInteger){
    mulmod(a, xisubxim.re, rid),
    mulmod(a, xisubxim.imag, rid),
  }; 

  return;
}

__device__ void idgttransform(
  GaussianInteger* data,
  const int cid,
  const int rid,
  const int stride,
  const int N,
  const int nresidues,
  const int m,
  const int log2k,
  const uint64_t *invgjk){

  // Indexes
  const int j = cid * 2 * m / N;
  const int i = j + (cid % (N/(2*m)))*2*m;
  const int xi_index = i;
  const int xim_index = i + m;

  // Coefs
  GaussianInteger xi = data[xi_index];
  GaussianInteger xim = data[xim_index];
  GaussianInteger new_xi, new_xim;

  // IDGT
  const uint64_t a = invgjk[(rid * log2k + stride) * N + j];
  const GaussianInteger axim = (GaussianInteger){
    mulmod(a, xim.re, rid),
    mulmod(a, xim.imag, rid)
  };
 
  new_xi = GIAdd(xi, axim, rid); 
  new_xim = GISub(xi, axim, rid); 

  // Write the result
  data[xi_index] = new_xi;
  data[xim_index] = new_xim;

  return;
}

__device__ void doDGT(
  GaussianInteger* data,
  int tid,
  int rid,
  const int nresidues,
  const int N,
  const int log2k,
  const uint64_t *gjk){

  for(int stride = 0; stride < log2k; stride++){
    int m = N / (2<<stride);
    dgttransform(
      data,
      tid,
      rid,
      stride,
      N,
      nresidues,
      m,
      log2k,
      gjk);
    __syncthreads();
  }
}



__device__ void doIDGT(
  GaussianInteger* data,
  int tid,
  int rid,
  const int nresidues,
  const int N,
  const int log2k,
  const uint64_t *invgjk){

  for(int stride = log2k - 1; stride >= 0; stride--){
    int m = N / (2<<stride);
    idgttransform(
      data,
      tid,
      rid,
      stride,
      N,
      nresidues,
      m,
      log2k,
      invgjk);
    __syncthreads();
  }

}

/*DGT "along the rows" 
data is treated as a Na x Nb matrix (row-major)
perform Nc DGTs of size 2 * blockDim.x "along the rows", I mean data is 
read through rows.

This kernel applies the DGT on columns. The outcome is written transposed.
Because of this we need to read and write on different arrays, otherwise 
we would create a race condition.

The template parameter is used to instantiate s_data.
*/
template <int Nblock>
__global__ void hdgt_tr(
  GaussianInteger* odata,
  GaussianInteger* idata,
  const int nresidues,
  const int rid_offset,
  const int log2k,
  const GaussianInteger *nthroot,
  const uint64_t *gjk){

  const int Nr = blockDim.x * 2; // Number of columns
  const int Nc = gridDim.x / nresidues; // Number of rows
  const int rid = (blockIdx.x  / Nc) + rid_offset;
  const int index_offset  = (rid - rid_offset) * Nc * Nr; // offset to target a particular residue
 
  const int in_xi_index   = threadIdx.x                * Nc + blockIdx.x % Nc;
  const int in_xim_index  = (threadIdx.x + blockDim.x) * Nc + blockIdx.x % Nc;
  const int out_xi_index  = threadIdx.x                     + blockIdx.x * Nr;
  const int out_xim_index = (threadIdx.x + blockDim.x)      + blockIdx.x * Nr; // write transposed
 
  __shared__ GaussianInteger s_data[Nblock];

  s_data[threadIdx.x]              = idata[in_xi_index + index_offset];
  s_data[threadIdx.x + blockDim.x] = idata[in_xim_index + index_offset];

  // Twist
  s_data[threadIdx.x]              =  GIMul(
    s_data[threadIdx.x],
    nthroot[in_xi_index + index_offset], rid);
  
  s_data[threadIdx.x + blockDim.x] =  GIMul(
    s_data[threadIdx.x + blockDim.x],
    nthroot[in_xim_index + index_offset], rid);

  // Twist
  __syncthreads();

  // DGT
  doDGT(s_data, threadIdx.x, rid, nresidues,Nblock, log2k, gjk);

  // Outcome
  odata[out_xi_index]  = s_data[threadIdx.x];
  odata[out_xim_index] = s_data[threadIdx.x + blockDim.x];
}

template <int Nblock>
__global__ void hdgt_tr(
  GaussianInteger* odata,
  GaussianInteger* idata,
  const int nresidues,
  const int rid_offset,
  const int log2k,
  const uint64_t *gN,
  const uint64_t *gjk){

  const int Nr = blockDim.x * 2; // Number of columns
  const int Nc = gridDim.x / nresidues; // Number of rows
  const int rid = (blockIdx.x  / Nc) + rid_offset;
  const int index_offset  = (rid - rid_offset) * Nc * Nr; // offset to target a particular residue
 
  const int in_xi_index   = threadIdx.x                * Nc + blockIdx.x % Nc;
  const int in_xim_index  = (threadIdx.x + blockDim.x) * Nc + blockIdx.x % Nc;
  const int out_xi_index  = threadIdx.x                     + blockIdx.x * Nr;
  const int out_xim_index = (threadIdx.x + blockDim.x)      + blockIdx.x * Nr; // write transposed
 
  __shared__ GaussianInteger s_data[Nblock];

  s_data[threadIdx.x]              = idata[in_xi_index + index_offset];
  s_data[threadIdx.x + blockDim.x] = idata[in_xim_index + index_offset];

  // Twist
  s_data[threadIdx.x] = (GaussianInteger){
    mulmod(gN[in_xi_index + index_offset], s_data[threadIdx.x].re, rid),
    mulmod(gN[in_xi_index + index_offset], s_data[threadIdx.x].imag, rid)
  };
  s_data[threadIdx.x + blockDim.x] = (GaussianInteger){
    mulmod(gN[in_xim_index + index_offset], s_data[threadIdx.x + blockDim.x].re, rid),
    mulmod(gN[in_xim_index + index_offset], s_data[threadIdx.x + blockDim.x].imag, rid)
  };

  // Twist
  __syncthreads();

  // DGT
  doDGT(s_data, threadIdx.x, rid, nresidues,Nblock, log2k, gjk);

  // Outcome
  odata[out_xi_index]  = s_data[threadIdx.x];
  odata[out_xim_index] = s_data[threadIdx.x + blockDim.x];
}

/*IDGT "along the rows"

data is treated as a Na x Nb matrix (row-major)
perform Nc IDGTs of size 2 * blockDim.x "along the columns".

This kernel applies the IDGT on rows. 
*/
template <int Nblock>
__global__ void hidgt(
  GaussianInteger* out_data,
  GaussianInteger* data,
  const int nresidues,
  const int rid_offset,
  const int log2k,
  const uint64_t *ginvN,
  const uint64_t *invgjk){

  const int rid = (blockIdx.x / (gridDim.x / nresidues)) + rid_offset;
  const int index_offset = blockIdx.x * Nblock; // offset to choose the residue to read
  
  const int xi_index = threadIdx.x;
  const int xim_index = threadIdx.x + blockDim.x;
  
  __shared__ GaussianInteger s_data[Nblock];

  s_data[xi_index]  = data[xi_index  + index_offset];
  s_data[xim_index] = data[xim_index + index_offset];
  
  __syncthreads();
  
  // IDGT
  doIDGT(s_data, threadIdx.x, rid, nresidues, Nblock, log2k, invgjk);

  // Untwist
  s_data[xi_index] = (GaussianInteger){
    mulmod(ginvN[xi_index + index_offset], s_data[xi_index].re, rid),
    mulmod(ginvN[xi_index + index_offset], s_data[xi_index].imag, rid)
  };
  s_data[xim_index] = (GaussianInteger){
    mulmod(ginvN[xim_index + index_offset], s_data[xim_index].re, rid),
    mulmod(ginvN[xim_index + index_offset], s_data[xim_index].imag, rid)
  };

  // Outcome
  out_data[xi_index + index_offset]  = s_data[xi_index];
  out_data[xim_index + index_offset] = s_data[xim_index];
}

/*IDGT "along the rows" 
data is treated as a Na x Nb matrix (row-major)
perform Nc IDGTs of size 2 * blockDim.x "along the rows".

This kernel applies the IDGT on columns. Different than hdgt_tr, the outcome is
NOT written transposed.
*/
template <int Nblock>
__global__ void hidgt_tr(
  GaussianInteger* odata,
  GaussianInteger* idata,
  const int nresidues,
  const int rid_offset,
  const int log2k,
  const GaussianInteger *invnthroot,
  const uint64_t *invgjk){

  const int Nr = blockDim.x * 2; // Number of columns
  const int Nc = gridDim.x / nresidues; // Number of rows
  const int rid = (blockIdx.x / Nc) + rid_offset;
  const int index_offset = (rid - rid_offset) * Nc * Nr; // offset to specify the residue to read

  const int in_xi_index  =  threadIdx.x                * Nc + blockIdx.x % Nc;
  const int in_xim_index = (threadIdx.x + blockDim.x)  * Nc + blockIdx.x % Nc;
  const int out_xi_index  =  in_xi_index;
  const int out_xim_index =  in_xim_index;
 
  __shared__ GaussianInteger s_data[Nblock];

  s_data[threadIdx.x]                = idata[in_xi_index + index_offset];
  s_data[(threadIdx.x + blockDim.x)] = idata[in_xim_index + index_offset];

  __syncthreads();
    
  doIDGT(s_data, threadIdx.x, rid, nresidues, Nblock, log2k, invgjk);

  // Untwist
  s_data[threadIdx.x]              =  GIMul(
    s_data[threadIdx.x],
    invnthroot[in_xi_index + index_offset], rid);
  
  s_data[threadIdx.x + blockDim.x] =  GIMul(
    s_data[threadIdx.x + blockDim.x],
    invnthroot[in_xim_index + index_offset], rid);

  // Outcome
  odata[out_xi_index  + index_offset]  = s_data[threadIdx.x];
  odata[out_xim_index + index_offset] = s_data[threadIdx.x + blockDim.x];
}

// Standard gentleman-sande DGT
template <int N>
__global__ void dgt(
  GaussianInteger* data,
  const int nresidues,
  const int rid_offset,
  const int log2k,
  const uint64_t *gjk,
  const GaussianInteger *nthroots){

  const int rid = blockIdx.x + rid_offset;
  const int index_offset = blockIdx.x * N; // rid_offset does not affect this
  __shared__ GaussianInteger s_data[N];

  s_data[threadIdx.x] = data[threadIdx.x + index_offset];
  s_data[threadIdx.x + (N >> 1)] = data[threadIdx.x + (N >> 1) + index_offset];

  __syncthreads();

  // Twist the folded signal
  s_data[threadIdx.x]            = GIMul(s_data[threadIdx.x],             nthroots[threadIdx.x         + rid * N], rid); // Twist
  s_data[threadIdx.x + (N >> 1)] = GIMul(s_data[threadIdx.x + (N >> 1)],  nthroots[(threadIdx.x + (N >> 1)) + rid * N], rid); // Twist

  // DGT
  doDGT(s_data, threadIdx.x, rid, nresidues, N, log2k, gjk);

  // Outcome
  data[threadIdx.x + index_offset] = s_data[threadIdx.x];
  data[threadIdx.x + (N >> 1) + index_offset] = s_data[threadIdx.x + (N >> 1)];
}

// Standard cooley-tukey IDGT
template <int N>
__global__ void idgt(
  GaussianInteger* data,
  const int nresidues,
  const int rid_offset,
  const int log2k,
  const uint64_t *invgjk,
  const GaussianInteger *invnthroots){

  const int rid = blockIdx.x + rid_offset;
  const int index_offset = (rid - rid_offset) * N; // rid_offset does not affect this
  __shared__ GaussianInteger s_data[N];

  s_data[threadIdx.x] = data[threadIdx.x + index_offset];
  s_data[threadIdx.x + (N >> 1)] = data[threadIdx.x + (N >> 1) + index_offset];
  __syncthreads();
  
  // IDGT
  doIDGT(s_data, threadIdx.x, rid, nresidues, N, log2k, invgjk);

  // "Untwist" the folded signal
  s_data[threadIdx.x]       = GIMul(s_data[threadIdx.x],        invnthroots[threadIdx.x         + rid * N], rid); // "Untwist"
  s_data[threadIdx.x + (N >> 1)] = GIMul(s_data[threadIdx.x + (N >> 1)],  invnthroots[(threadIdx.x + (N >> 1)) + rid * N], rid); // "Untwist"

  // Outcome
  data[threadIdx.x + index_offset]       = s_data[threadIdx.x];
  data[threadIdx.x + (N >> 1) + index_offset] = s_data[threadIdx.x + (N >> 1)];
}

__host__ void DGTEngine::execute_dgt(
  Context *ctx,
  poly_t* p,
  const dgt_direction direction){


  assert(p->state != NONINITIALIZED);

  poly_states new_state = (direction == FORWARD? TRANSSTATE : RNSSTATE);
  if(p->state == new_state)
    return;
  
  // if(direction == FORWARD)
  //   std::cout << "FORWARD DGT" << endl;
  // else
  //   std::cout << "INVERSE DGT" << endl;

  GaussianInteger *data = p->d_coefs;
  poly_bases base = p->base;

  const int nresidues = CUDAEngine::get_n_residues(base);
  int rid_offset;

  switch(base){
    case QBase:
    case QBBase:
    rid_offset = 0;
    break;
    case BBase:
    rid_offset = CUDAEngine::RNSPrimes.size();
    break;
    case TBase:
    return;
    default:
    throw std::runtime_error("Unknown base");
  }

  // \todo Assert N is a power of 2
  runDGT(CUDAEngine::N);  

  p->state = new_state;
}

__global__ void add_dgt(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const int N,
  const int nresidues,
  const int rid_offset){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N + rid_offset;

  if(tid < N * nresidues)
    c_data[tid] = GIAdd(a_data[tid], b_data[tid], rid);
  
}


__host__ void DGTEngine::execute_add_dgt(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const int base,
  const cudaStream_t stream){

  const int N = CUDAEngine::N;
  const int nresidues = CUDAEngine::get_n_residues(base);
  int rid_offset;

  rid_offset = (base == BBase) * CUDAEngine::RNSPrimes.size();

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  add_dgt<<< gridDim, blockDim, 0, stream>>>(
    c_data,
    a_data,
    b_data,
    N,
    nresidues,
    rid_offset);
}

__global__ void sub_dgt(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const int N,
  const int nresidues,
  const int rid_offset){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N + rid_offset;

  if(tid < N * nresidues)
    c_data[tid] = GISub(a_data[tid], b_data[tid], rid);
}


__host__ void DGTEngine::execute_sub_dgt(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const int base,
  const cudaStream_t stream){

  const int N = CUDAEngine::N;
  const int nresidues = CUDAEngine::get_n_residues(base);
  int rid_offset;
  
  rid_offset = (base == BBase) * CUDAEngine::RNSPrimes.size();

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  sub_dgt<<< gridDim, blockDim, 0, stream>>>(
    c_data,
    a_data,
    b_data,
    N,
    nresidues,
    rid_offset);
}

__global__ void double_add_dgt(
  GaussianInteger *c1_data,
  const GaussianInteger *a1_data,
  const GaussianInteger *b1_data,
  GaussianInteger *c2_data,
  const GaussianInteger *a2_data,
  const GaussianInteger *b2_data,
  const int N,
  const int nresidues,
  const int rid_offset){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N + rid_offset;

  if(tid < N * nresidues){
    c1_data[tid] = GIAdd(a1_data[tid], b1_data[tid], rid);
    c2_data[tid] = GIAdd(a2_data[tid], b2_data[tid], rid);
  }
}

__host__ void DGTEngine::execute_double_add_dgt(
  GaussianInteger *c1_data,
  const GaussianInteger *a1_data,
  const GaussianInteger *b1_data,
  GaussianInteger *c2_data,
  const GaussianInteger *a2_data,
  const GaussianInteger *b2_data,
  const int base,
  const cudaStream_t stream){

  const int N = CUDAEngine::N;
  const int nresidues = CUDAEngine::get_n_residues(base);
  int rid_offset;

  // switch(base){
  //   case QBase:
  //   case QBBase:
  //   rid_offset = 0;
  //   break;
  //   case BBase:
  //   rid_offset = CUDAEngine::RNSPrimes.size();
  //   break;
  //   case TBase:
  //   return;
  //   default:
  //   throw std::runtime_error("Unknown base");
  // }
  rid_offset = (base == BBase) * CUDAEngine::RNSPrimes.size();

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  double_add_dgt<<< gridDim, blockDim, 0, stream>>>(
    c1_data,
    a1_data,
    b1_data,
    c2_data,
    a2_data,
    b2_data,
    N,
    nresidues,
    rid_offset);
}



__global__ void dr2_dgt(
  GaussianInteger *ct21, // Outcome
  GaussianInteger *ct22, // Outcome
  GaussianInteger *ct23, // Outcome
  const GaussianInteger *ct01, // Operand 1
  const GaussianInteger *ct02, // Operand 1
  const GaussianInteger *ct11, // Operand 2
  const GaussianInteger *ct12, // Operand 2
  const int N,
  const int nresidues,
  const int rid_offset){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N + rid_offset;

  if(tid < N * nresidues){
    GaussianInteger local_ct01 = ct01[tid], local_ct02 = ct02[tid];
    GaussianInteger local_ct11 = ct11[tid], local_ct12 = ct12[tid];

    ct21[tid] = GIMul(local_ct01, local_ct11, rid);

    ct22[tid] = GIAdd(
      GIMul(local_ct01, local_ct12, rid),
      GIMul(local_ct02, local_ct11, rid),
      rid);

    ct23[tid] = GIMul(local_ct02, local_ct12, rid);
  }
}

__host__ void DGTEngine::execute_dr2_dgt(
  GaussianInteger *ct21, // Outcome
  GaussianInteger *ct22, // Outcome
  GaussianInteger *ct23, // Outcome
  const GaussianInteger *ct01, // Operand 1
  const GaussianInteger *ct02, // Operand 1
  const GaussianInteger *ct11, // Operand 2
  const GaussianInteger *ct12, // Operand 2
  const int base,
  const cudaStream_t stream){

  const int N = CUDAEngine::N;
  const int nresidues = CUDAEngine::get_n_residues(base);
  const int rid_offset = (base == BBase) * CUDAEngine::RNSPrimes.size();

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  dr2_dgt<<< gridDim, blockDim, 0, stream>>>(
    ct21,
    ct22,
    ct23,
    ct01,
    ct02,
    ct11,
    ct12,
    N,
    nresidues,
    rid_offset);
}

template <class T>
__global__ void mul_dgt(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const T *b_data,
  const int N,
  const int nresidues,
  const int rid_offset){

}

template <>
__global__ void mul_dgt(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const int N,
  const int nresidues,
  const int rid_offset){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N + rid_offset;

  if(tid < N * nresidues)
    c_data[tid] = GIMul(a_data[tid], b_data[tid], rid);
}

template <>
__global__ void mul_dgt(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const uint64_t *b_data,
  const int N,
  const int nresidues,
  const int rid_offset){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N + rid_offset;

  if(tid < N * nresidues)
    c_data[tid] = (GaussianInteger){
      mulmod(b_data[tid], a_data[tid].re, rid),
      mulmod(b_data[tid], a_data[tid].imag, rid)
    };
}


__host__ void DGTEngine::execute_mul_dgt_gi(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const int base,
  const cudaStream_t stream){

  const int N = CUDAEngine::N;
  const int nresidues = CUDAEngine::get_n_residues(base);
  int rid_offset;

  switch(base){
    case QBase:
    case QBBase:
    rid_offset = 0;
    break;
    case BBase:
    rid_offset = CUDAEngine::RNSPrimes.size();
    break;
    case TBase:
    return;
    default:
    throw std::runtime_error("Unknown base");
  }

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  mul_dgt<GaussianInteger><<< gridDim, blockDim, 0, stream>>>(
    c_data,
    a_data,
    b_data,
    N,
    nresidues,
    rid_offset);
}

__host__ void DGTEngine::execute_mul_dgt_u64(
  GaussianInteger *c_data,
  const GaussianInteger *a_data,
  const uint64_t *b_data,
  const int base,
  const cudaStream_t stream){

  const int N = CUDAEngine::N;
  const int nresidues = CUDAEngine::get_n_residues(base);
  int rid_offset;

  switch(base){
    case QBase:
    case QBBase:
    rid_offset = 0;
    break;
    case BBase:
    rid_offset = CUDAEngine::RNSPrimes.size();
    break;
    case TBase:
    return;
    default:
    throw std::runtime_error("Unknown base");
  }

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  mul_dgt<uint64_t><<< gridDim, blockDim, 0, stream>>>(
    c_data,
    a_data,
    b_data,
    N,
    nresidues,
    rid_offset);
}

__global__ void muladd_dgt(
  GaussianInteger *d_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const GaussianInteger *c_data,
  const int N,
  const int nresidues,
  const int rid_offset){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N + rid_offset;

  if(tid < N * nresidues)
    d_data[tid] = GIAdd(
      GIMul(a_data[tid], b_data[tid], rid),
      c_data[tid],
      rid
      );
}


__host__ void DGTEngine::execute_muladd_dgt(
  GaussianInteger *d_data,
  const GaussianInteger *a_data,
  const GaussianInteger *b_data,
  const GaussianInteger *c_data,
  const int base,
  const cudaStream_t stream){

  const int N = CUDAEngine::N;
  const int nresidues = CUDAEngine::get_n_residues(base);
  int rid_offset;

  switch(base){
    case QBase:
    case QBBase:
    rid_offset = 0;
    break;
    case BBase:
    rid_offset = CUDAEngine::RNSPrimes.size();
    break;
    case TBase:
    return;
    default:
    throw std::runtime_error("Unknown base");
  }

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  muladd_dgt<<< gridDim, blockDim, 0, stream>>>(
    d_data,
    a_data,
    b_data,
    c_data,
    N,
    nresidues,
    rid_offset);
}

//
int ilog2(int n){
  return (int)floor(log2(n));
}

/* Pre-compute the powers of g for a particular n
  writes the output to o_gjk and the inverses to o_invgjk 
  (both in device's global memory)*/
void compute_gjk(
  uint64_t *o_gjk,
  uint64_t *o_invgjk,
  int signalsize){

  // Alloc
  uint64_t *h_gjk = (uint64_t*) calloc ( 
    CUDAEngine::get_n_residues(QBBase) * ilog2(signalsize) * signalsize, 
    sizeof(uint64_t) );
  uint64_t *h_invgjk = (uint64_t*) calloc ( 
    CUDAEngine::get_n_residues(QBBase) * ilog2(signalsize) * signalsize, 
    sizeof(uint64_t) );

  // Compute gjk and its inverse for each coprime
  for(
    int rid = 0;
    rid < CUDAEngine::get_n_residues(QBBase);
    rid++){

    uint64_t p = COPRIMES_BUCKET[rid];    

    // Assertions
    assert((p-1) % (2*signalsize) == 0); // k | (p-1)
    const uint64_t n = (p-1)/(2*signalsize);
    const uint64_t g = fast_pow((uint64_t)PROOTS[p], n, rid);

    // Pre-compute g^j
    for(int stride = 0; stride < ilog2(signalsize); stride++){
      for(int j = 0; j < signalsize; j++){
        h_gjk[rid * ilog2(signalsize) * signalsize + stride * signalsize + j] = 
          fast_pow(
              g,
              j * ((signalsize * 2) >> (ilog2(signalsize) - stride)),
              rid
            ) % p;
      }
    }

  // Pre-compute the moduler inverses of g^j
   for(int stride = ilog2(signalsize)-1; stride >= 0; stride--){
      for(int j = 0; j < signalsize; j++){
        h_invgjk[rid * ilog2(signalsize) * signalsize + stride * signalsize + j] = 
          fast_pow(
              g,
              (signalsize * 2 - j) * ((signalsize * 2) >> (ilog2(signalsize) - stride)),
              rid
            ) % p;
      }
    }

  }

  // Check
  for(int z = 0; z < CUDAEngine::get_n_residues(QBBase); z++){
    uint64_t p = COPRIMES_BUCKET[z];    
    for(int i = 0; i < ilog2(signalsize); i++)
      for(int j = 0; j < signalsize; j++){
        uint64_t r = mulmod(
          h_gjk[z * ilog2(signalsize) * signalsize + i * signalsize + j],
          h_invgjk[z * ilog2(signalsize) * signalsize + i * signalsize + j],
          z);
          if(r != 1){
            Logger::getInstance()->log_error("I can't confirm correctness for h_gjk * h_invgjk");
            Logger::getInstance()->log_error((std::string("Residue ") + std::to_string(p)).c_str());
          }
      }
  }

  // Copy to global memory
  cudaMemcpy(
    o_gjk,
    h_gjk,
    CUDAEngine::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError()

  cudaMemcpy(
    o_invgjk,
    h_invgjk,
    CUDAEngine::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError()

  // Release temporary memory
  free(h_gjk);
  free(h_invgjk);
}

/**
 * @brief      Calculates the nthroot.
 *
 * Pre-compute the powers of the n-th root for a particular n and
 * writes the output to o_nthroots and the inverses to o_invnthroots 
 * (both in device's global memory).
 * 
 * @param[out]      o_d_nthroots     The powers of the nthroot in device's memory.
 * @param[out]      o_d_invnthroots  The powers of the inverse of nthroot in device's memory.
 * @param[out]      o_h_nthroots     The powers of the nthroot in host's memory.
 * @param[out]      o_h_invnthroots  The powers of the inverse of nthroot in host's memory.
 * @param[in]       signalsize       The signalsize
 */
void compute_nthroot(
  GaussianInteger *o_d_nthroots,
  GaussianInteger *o_d_invnthroots,
  GaussianInteger *o_h_nthroots,
  GaussianInteger *o_h_invnthroots,
  int signalsize,
  bool isHDGT = false){

  // Alloc
  GaussianInteger *h_nthroots = (GaussianInteger*) calloc ( 
    signalsize * CUDAEngine::get_n_residues(QBBase), 
    sizeof(GaussianInteger));
  GaussianInteger *h_invnthroots = (GaussianInteger*) calloc ( 
    signalsize * CUDAEngine::get_n_residues(QBBase), 
    sizeof(GaussianInteger));

  /**
   Verifies if all roots are valid signalsize-th primitive roots and if the inverses
   are really modular inverses of those
   */
  for(
    int rid = 0;
    rid < CUDAEngine::get_n_residues(QBBase);
    rid++){
    uint64_t p = COPRIMES_BUCKET[rid];    

    // Do the values really exist?
    if(NTHROOT[p].count(signalsize) < 1 || INVNTHROOT[p].count(signalsize) < 1)
      throw std::runtime_error(std::string("ERROR! I don't know the ") + 
      std::to_string(signalsize) +
      "th-primitive root of i");

    assert(GIFastPow(NTHROOT[p][signalsize], signalsize, rid) == ((GaussianInteger){0, 1}));
    assert(GIMul(NTHROOT[p][signalsize], INVNTHROOT[p][signalsize], rid) == ((GaussianInteger){1, 0}));

  }

  // Compute the powers of the root
  for(
    int rid = 0;
    rid < CUDAEngine::get_n_residues(QBBase);
    rid++)
    for(int cid = 0; cid < signalsize; cid++){
      h_nthroots   [cid + rid * signalsize] = GIFastPow(NTHROOT   [COPRIMES_BUCKET[rid]][signalsize], cid, rid);
      if(!isHDGT) // Normal DGT
        h_invnthroots[cid + rid * signalsize] = GIMul(
        GIFastPow(INVNTHROOT[COPRIMES_BUCKET[rid]][signalsize], cid, rid),
        (GaussianInteger){ conv<uint64_t>(
            NTL::InvMod(to_ZZ(signalsize), to_ZZ(COPRIMES_BUCKET[rid])) // Embed the scaling factor
          ),
          0},
        rid
        );
      else // Hierarchical DGT
        h_invnthroots[cid + rid * signalsize] = GIFastPow(INVNTHROOT[COPRIMES_BUCKET[rid]][signalsize], cid, rid);
    }
  
  // Copy the powers of the root
  cudaMemcpy(
    o_d_nthroots,
    h_nthroots,
    signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger),
    cudaMemcpyHostToDevice);
  cudaCheckError()
  cudaMemcpy(
    o_d_invnthroots,
    h_invnthroots,
    signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger),
    cudaMemcpyHostToDevice);
  cudaCheckError()
  memcpy(
    o_h_nthroots,
    h_nthroots,
    signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger));
  memcpy(
    o_h_invnthroots,
    h_invnthroots,
    signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger));

  // Release temporary memory
  free(h_nthroots);
  free(h_invnthroots);
}

__host__ void DGTEngine::init(){
  if(
    (CUDAEngine::N == 1024 && !ENABLE_HDGT_1024) || 
    (CUDAEngine::N == 2048 && !ENABLE_HDGT_2048)
    )
    std::cout << "WARNING: Running in DGT-II mode." << std::endl;

  std::ostringstream os;
  std::ostringstream os_debug;

  // Common assertions
  assert(CUDAEngine::is_init);
  assert(CUDAEngine::N > 0);
  
  // Alloc
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaCheckError();

  // Precompute the size of each coprime
  for(
    int rid = 0;
    rid < CUDAEngine::get_n_residues(QBBase);
    rid++){
    uint64_t p = COPRIMES_BUCKET[rid];    
    RNSCoprimes_NumBits[rid] = NTL::NumBits(to_ZZ(p));
  }
  cudaMemcpyToSymbol(
    d_RNSCoprimes_NumBits,
    RNSCoprimes_NumBits,
    CUDAEngine::get_n_residues(QBBase) * sizeof(int),
    0,
    cudaMemcpyHostToDevice
  );

  // bitreverse
  // We use CUDAEngine::N as upper bound
  int* h_bitreversalmap_Na = (int*) calloc (CUDAEngine::N, sizeof(int)); 
  int* h_bitreversalmap_Nb = (int*) calloc (CUDAEngine::N, sizeof(int));

  // powers of g
  uint64_t *h_gN = (uint64_t*) calloc ( 
    CUDAEngine::N * CUDAEngine::get_n_residues(QBBase), 
    sizeof(uint64_t));
  uint64_t *h_ginvN = (uint64_t*) calloc ( 
    CUDAEngine::N * CUDAEngine::get_n_residues(QBBase), 
    sizeof(uint64_t));
  
  cudaMalloc(
    (void**)&d_gN,
    CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t)
    );
  cudaCheckError();

  cudaMalloc(
    (void**)&d_ginvN,
    CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t)
    );
  cudaCheckError();

  // Compute the constant factors required by Barrett reduction
  RR::SetPrecision(2048);
  RR::SetOutputPrecision(100);
  for(
    int rid = 0;
    rid < CUDAEngine::get_n_residues(QBBase);
    rid++){
    uint64_t p = COPRIMES_BUCKET[rid];    

    RR mu_dividend = to_RR(NTL::power(to_ZZ(4),NumBits(p))); // 2 ^ {ceil(log(p,2)) * 2}
    RR mu_divisor = to_RR(p);
    h_BARRETT_MU[rid] = conv<uint64_t>(floor(mu_dividend / mu_divisor));
  }

  cudaMemcpyToSymbol(
    d_BARRETT_MU,
    h_BARRETT_MU,
    CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t),
    0,
    cudaMemcpyHostToDevice
  );

  /* If hierarchical DGT we need to pre-compute these powers for different 
  signl sizes (64, 128, ...). Otherwise, we only need one 
  signal size = CUDAEngine::N */
  std::vector<int> signalsizes {};
  bool isHDGT = false;
  int Na = -1, Nb = -1;
  switch(CUDAEngine::N){
    case 1024:
    Na = DGTFACT1024;
    Nb = DGTFACT1024;

    isHDGT = ENABLE_HDGT_1024;
    break;
    case 2048:
    Na = DGTFACT2048a;
    Nb = DGTFACT2048b;

    isHDGT = ENABLE_HDGT_2048;
    break;
    case 4096:
    Na = DGTFACT4096;
    Nb = DGTFACT4096;

    isHDGT = true;
    break;
    case 8192:
    Na = DGTFACT8192a;
    Nb = DGTFACT8192b;

    isHDGT = true;
    break;
    case 16384:
    Na = DGTFACT16384;
    Nb = DGTFACT16384;

    isHDGT = true;
    break;
    case 32768:
    Na = DGTFACT32768a;
    Nb = DGTFACT32768b;

    isHDGT = true;
    break;
  }
  if(isHDGT){
    signalsizes.push_back(Na);
    if(Nb != Na)
      signalsizes.push_back(Nb);
  }
  signalsizes.push_back(CUDAEngine::N);
  for(unsigned int i = 0; i < signalsizes.size(); i++){
    int signalsize = signalsizes[i];

    uint64_t *gjk_aux;

    cudaMalloc(
      (void**)&gjk_aux,
      CUDAEngine::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t)
      );
    cudaCheckError();
    d_gjk[signalsize] = gjk_aux;

    cudaMalloc(
      (void**)&gjk_aux,
      CUDAEngine::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t)
      );
    cudaCheckError();
    d_invgjk[signalsize] = gjk_aux;

    GaussianInteger *nthroot_aux;
    cudaMalloc(
      (void**)&nthroot_aux,
      signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger)
      );
    cudaCheckError();
    d_nthroot[signalsize] = nthroot_aux;

    cudaMalloc(
      (void**)&nthroot_aux,
      signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger)
      );
    cudaCheckError();
    d_invnthroot[signalsize] = nthroot_aux;

    h_nthroot[signalsize] = (GaussianInteger*) malloc(
      signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger));
    h_invnthroot[signalsize] = (GaussianInteger*) malloc(
      signalsize * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger));

    // pre-compute
    compute_gjk(
      d_gjk[signalsize],
      d_invgjk[signalsize],
      signalsize);
    compute_nthroot(
      d_nthroot[signalsize],
      d_invnthroot[signalsize],
      h_nthroot[signalsize],
      h_invnthroot[signalsize],
      signalsize,
      isHDGT);
  }

  if(isHDGT){
    // Compute the bitreverse mapping
    // init
    int j = 0;
    for(int i = 1; i < Na; i++)
      h_bitreversalmap_Na[i] = i;
    for(int i = 1; i < Nb; i++)
      h_bitreversalmap_Nb[i] = i;
    // compute
    for(int i = 1; i < Na; i++){
      int b = Na >> 1;
      while(j >= b){
        j -= b;
        b >>= 1;
      }
      j += b;
      if(j > i){
        h_bitreversalmap_Na[i] = j;
        h_bitreversalmap_Na[j] = i;
      }
    }
    j = 0;
    for(int i = 1; i < Nb; i++){
      int b = Nb >> 1;
      while(j >= b){
        j -= b;
        b >>= 1;
      }
      j += b;
      if(j > i){
        h_bitreversalmap_Nb[i] = j;
        h_bitreversalmap_Nb[j] = i;
      }
    }

    // Compute the powers of the nth primitive root of unity for N = Na*Nb
    for(
      int rid = 0;
      rid < CUDAEngine::get_n_residues(QBBase);
      rid++){
      uint64_t p = COPRIMES_BUCKET[rid];
      uint64_t n = (p-1)/(CUDAEngine::N);
      assert((p-1) % CUDAEngine::N == 0);

      uint64_t g = fast_pow((uint64_t)PROOTS[p], n, rid);
      uint64_t g_inv = conv<uint64_t>(NTL::InvMod(to_ZZ(g), to_ZZ(p)));

      assert(mulmod(g, g_inv, rid) % p == 1);
      assert(Na && Nb);
      for(int cid = 0; cid < CUDAEngine::N; cid++){
        h_gN   [cid + rid * CUDAEngine::N] = fast_pow(
          g,
          (cid / Nb) * h_bitreversalmap_Nb[cid % Nb],
          rid);
        h_ginvN[cid + rid * CUDAEngine::N] = mulmod(
            fast_pow(
            g_inv,
            h_bitreversalmap_Nb[cid / Na] * (cid % Na),
            rid),
            conv<uint64_t>(NTL::InvMod(to_ZZ(Na * Nb), to_ZZ(COPRIMES_BUCKET[rid]))), // Embed the scaling
            rid);
      }
    }

    // Copy the powers of the root
    cudaMemcpyAsync(
      d_gN,
      h_gN,
      CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t),
      cudaMemcpyHostToDevice,
      stream
      );
    cudaCheckError()
    cudaMemcpyAsync(
      d_ginvN,
      h_ginvN,
      CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t),
      cudaMemcpyHostToDevice,
      stream
      );
    cudaCheckError()
  }


  // Synchronize
  cudaStreamSynchronize(stream);
  cudaCheckError();
  cudaStreamDestroy(stream);
  cudaCheckError();

  free(h_bitreversalmap_Na);
  free(h_bitreversalmap_Nb);
  free(h_gN);
  free(h_ginvN);

  is_init = true;

  ///////////////
  // Greetings //
  ///////////////
  os << "DGT initialized." << std::endl;
  Logger::getInstance()->log_info(os.str().c_str());
  Logger::getInstance()->log_debug(os_debug.str().c_str());
}

__host__ void DGTEngine::destroy(){
  if(!is_init)
    return;

  cudaDeviceSynchronize();
  cudaCheckError();

  cudaFree(d_gN);
  cudaCheckError();
  cudaFree(d_ginvN);
  cudaCheckError();

  for(auto const& x : d_gjk){
    cudaFree(x.second);
    cudaCheckError();
  }
  for(auto const& x : d_invgjk){
    cudaFree(x.second);
    cudaCheckError();
  }
  for(auto const& x : d_nthroot){
    cudaFree(x.second);
    cudaCheckError();
  }
  for(auto const& x : d_invnthroot){
    cudaFree(x.second);
    cudaCheckError();
  }
  for(auto const& x : h_nthroot)
    free(x.second);
  for(auto const& x : h_invnthroot)
    free(x.second);

  d_gjk.clear();
  d_invgjk.clear();
  d_nthroot.clear();
  d_invnthroot.clear();
  h_nthroot.clear();
  h_invnthroot.clear();

  is_init = false;
}
