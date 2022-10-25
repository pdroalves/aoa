#include <newckks/cuda/htrans/common.h>
#include <newckks/cuda/htrans/ntt.h>
#include <newckks/cuda/htrans/dgt.h>
#include <newckks/cuda/manager.h>
#include <newckks/coprimes.h>
#include <omp.h>
using namespace NTL;

extern __constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];
__constant__ int d_RNSCoprimes_NumBits[MAX_COPRIMES];
int RNSCoprimes_NumBits[MAX_COPRIMES];
__constant__ uint64_t d_BARRETT_MU[MAX_COPRIMES];
uint64_t h_BARRETT_MU[MAX_COPRIMES]; // Debug purposes only
bool COMMONEngine::is_init = false;
std::map<int, uint64_t*> COMMONEngine::d_gjk;
std::map<int, uint64_t*> COMMONEngine::d_invgjk;
engine_types COMMONEngine::engine_mode;

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

//
int ilog2(int n){
  return (int)floor(log2(n));
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

__host__ __device__ uint128_t add128(uint128_t x, uint128_t y, int rid){
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

__host__ __device__ inline void load_coprime(uint64_t *p, int *numbits, int rid){
#ifdef __CUDA_ARCH__
  *p = d_RNSCoprimes[rid];
  *numbits = d_RNSCoprimes_NumBits[rid];
#else
  *p = COPRIMES_BUCKET[rid];
  *numbits = RNSCoprimes_NumBits[rid];
#endif
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

////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////


/* Pre-compute the powers of g for a particular n
  writes the output to o_gjk and the inverses to o_invgjk 
  (both in device's global memory)*/
void compute_gjk(
  uint64_t *o_gjk,
  uint64_t *o_invgjk,
  int signalsize){

  // Alloc
  uint64_t *h_gjk = (uint64_t*) calloc ( 
    CUDAManager::get_n_residues(QBBase) * ilog2(signalsize) * signalsize, 
    sizeof(uint64_t) );
  uint64_t *h_invgjk = (uint64_t*) calloc ( 
    CUDAManager::get_n_residues(QBBase) * ilog2(signalsize) * signalsize, 
    sizeof(uint64_t) );

  // Compute gjk and its inverse for each coprime
  #pragma omp parallel for
  for(
    int rid = 0;
    rid < CUDAManager::get_n_residues(QBBase);
    rid++){

    uint64_t p = COPRIMES_BUCKET[rid];    

    // Assertions
    assert(signalsize > 0);
    assert((p-1) % (2*signalsize) == 0); // k | (p-1)
    const uint64_t n = (p-1)/(2*signalsize);
    const uint64_t g = fast_pow((uint64_t)PROOTS[p], n, rid);

    // Pre-compute g^j
    for(int stride = 0; stride < ilog2(signalsize); stride++)
      for(int j = 0; j < signalsize; j++){
        int idx = rid * ilog2(signalsize) * signalsize + stride * signalsize + j; 
        int n = j * ((signalsize * 2) >> (ilog2(signalsize) - stride));
        h_gjk[idx] = fast_pow(g, n, rid);
      }
    
   // Pre-compute the moduler inverses of g^j
   for(int stride = ilog2(signalsize)-1; stride >= 0; stride--)
      for(int j = 0; j < signalsize; j++){
        int idx = rid * ilog2(signalsize) * signalsize + stride * signalsize + j;
        int n = (signalsize * 2 - j) * ((signalsize * 2) >> (ilog2(signalsize) - stride));
        h_invgjk[idx] = fast_pow(g,n,rid);
      }
  }

  // Check
  #ifdef ENABLE_DEBUG_MACRO
  for(int z = 0; z < CUDAManager::get_n_residues(QBBase); z++){
    uint64_t p = COPRIMES_BUCKET[z];    
    for(int i = 0; i < ilog2(signalsize); i++)
      for(int j = 0; j < signalsize; j++){
        uint64_t r = mulmod(
          h_gjk[z * ilog2(signalsize) * signalsize + i * signalsize + j],
          h_invgjk[z * ilog2(signalsize) * signalsize + i * signalsize + j],
          z);
          if(r != 1){
            std::cout << "I can't confirm correctness for h_gjk * h_invgjk (received " << r << ")" << std::endl;
            std::cout << "Residue " << p << std::endl;
          }
      }
  }
  #endif

  // Copy to global memory
  cudaMemcpy(
    o_gjk,
    h_gjk,
    CUDAManager::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError()

  cudaMemcpy(
    o_invgjk,
    h_invgjk,
    CUDAManager::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError()

  // Release temporary memory
  free(h_gjk);
  free(h_invgjk);
}

void COMMONEngine::execute(
  Context *ctx,
  poly_t *data,
  const transform_directions direction){ // Forward or Inverse

  switch(engine_mode){
    case NTTTrans:
      NTTEngine::execute_ntt(ctx, data, direction);
    break;
    default:
    throw std::runtime_error("Unknown engine mode");
  }

}

///////////////////////////////////////////////////////////////////////////////


void COMMONEngine::execute_op(
  Context *ctx,
  uint64_t *c,
  uint64_t *a,
  uint64_t *b,
  const supported_operations OP,
  const poly_bases base){

  switch(engine_mode){
    case NTTTrans:
      NTTEngine::execute_op(ctx, c, a, b, OP, base);
    break;
    default:
    throw std::runtime_error("Unknown engine mode");
  }
}

void COMMONEngine::execute_op_by_uint(
  Context *ctx,
  uint64_t *c,
  uint64_t *a,
  const uint64_t b,
  const supported_operations OP,
  const poly_bases base){

  switch(engine_mode){
    case NTTTrans:
      NTTEngine::execute_op_by_uint(ctx, c, a, b, OP, base);
    break;
    default:
    throw std::runtime_error("Unknown engine mode");
  }
}

void COMMONEngine::execute_dualop(
  Context *ctx,
  uint64_t *c, uint64_t *a, uint64_t *b,
  uint64_t *f, uint64_t *d, uint64_t *e,
  const supported_operations OP,
  const poly_bases base){

  switch(engine_mode){
    case NTTTrans:
      NTTEngine::execute_dualop(ctx, c, a, b, f, d, e, OP, base);
    break;
    default:
    throw std::runtime_error("Unknown engine mode");
  }

}

void COMMONEngine::execute_seqop(
  Context *ctx,
  uint64_t *d,
  uint64_t *a, uint64_t *b, uint64_t *c,
  const supported_operations OP,
  const poly_bases base){

  switch(engine_mode){
    case NTTTrans:
      NTTEngine::execute_seqop(ctx, d, a, b, c, OP, base);
    break;
    default:
    throw std::runtime_error("Unknown engine mode");
  }

}

///////////////////////////////////////////////////////////////////////////////

void COMMONEngine::init(engine_types e){
  // Compute the constant factors required by Barrett reduction
  RR::SetPrecision(2048);
  RR::SetOutputPrecision(100);
  for(
    int rid = 0;
    rid < CUDAManager::get_n_residues(QBBase);
    rid++){
    uint64_t p = COPRIMES_BUCKET[rid];    

    RR mu_dividend = to_RR(NTL::power(to_ZZ(4),NumBits(p))); // 2 ^ {ceil(log(p,2)) * 2}
    RR mu_divisor = to_RR(p);
    h_BARRETT_MU[rid] = conv<uint64_t>(floor(mu_dividend / mu_divisor));
  }

  cudaMemcpyToSymbol(
    d_BARRETT_MU,
    h_BARRETT_MU,
    CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t),
    0,
    cudaMemcpyHostToDevice
  );

  // Precompute the size of each coprime
  for(
    int rid = 0;
    rid < CUDAManager::get_n_residues(QBBase);
    rid++){
    uint64_t p = COPRIMES_BUCKET[rid];    
    RNSCoprimes_NumBits[rid] = NTL::NumBits(to_ZZ(p));
  }
  cudaMemcpyToSymbol(
    d_RNSCoprimes_NumBits,
    RNSCoprimes_NumBits,
    CUDAManager::get_n_residues(QBBase) * sizeof(int),
    0,
    cudaMemcpyHostToDevice);

  is_init = true;

  #ifdef NDEBUG
  std::cout << "COMMONEngine initialized  = [" << e << "]" << std::endl;
  #endif

  // Init the transform
  switch(engine_mode){
    case NTTTrans:
      NTTEngine::init();
    break;
    default:
    throw std::runtime_error("Unknown engine mode");
  }

}

void COMMONEngine::destroy(){
  cudaDeviceSynchronize();
  cudaCheckError();

  switch(engine_mode){
    case NTTTrans:
      NTTEngine::destroy();
    break;
    default:
    throw std::runtime_error("Unknown engine mode");
  }

 for(auto const& x : d_gjk){
    cudaFree(x.second);
    cudaCheckError();
  }
  for(auto const& x : d_invgjk){
    cudaFree(x.second);
    cudaCheckError();
  }
  d_gjk.clear();
  d_invgjk.clear();

  is_init = false;
}