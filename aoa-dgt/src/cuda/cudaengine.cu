#include <AOADGT/cuda/cudaengine.h>
#include <functional>
#include <numeric>

ZZ CUDAEngine::RNSProduct;
ZZ CUDAEngine::RNSBProduct;
std::vector<uint64_t> CUDAEngine::RNSPrimes;
std::vector<ZZ> CUDAEngine::RNSMpi;
std::vector<uint64_t> CUDAEngine::RNSInvMpi;
std::vector<uint64_t> CUDAEngine::RNSBPrimes;
int CUDAEngine::N = 0;
int CUDAEngine::is_init = false;
uint64_t *CUDAEngine::RNSCoprimes;
uint64_t CUDAEngine::t;
int CUDAEngine::scalingfactor = -1;
int CUDAEngine::dnum = 1;

// \todo These shall be moved to inside CUDAEngine
uint32_t COPRIMES_BUCKET_SIZE; //!< The size of COPRIMES_55_BUCKET.

///////////////
// Main base //
///////////////
__constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];
__constant__ int      d_RNSQNPrimes;// Used primes

// HPS basis extension from B to Q and fast_conv_B_to_Q
__constant__ int      d_RNSBNPrimes;// Used primes
// rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
__constant__ uint64_t d_RNSBqi[MAX_COPRIMES_IN_A_BASE]; // Stores (B) \pmod qPi;
// HPS basis extension from B to Q and fast_conv_B_to_Q
__constant__ uint64_t d_RNSInvModBbi[MAX_COPRIMES_IN_A_BASE]; // Stores (B/bi)^-1 \pmod bi;

// HPS basis extension from B to Q and fast_conv_B_to_Q
uint64_t *d_RNSBbiqi; // Stores (B/bi) \pmod qi;
                                                                                   // 
// HPS basis extension  and fast_conv_Q_to_B
uint64_t *d_RNSInvModQqi; // Stores (Q/bi)^-1 \pmod qi; for each level

// HPS basis extension Q to B and fast_conv_Q_to_B
uint64_t *d_RNSQqibi; // Stores (Q/qi) \pmod bi; ; for each level

// CKKS rescale
uint64_t *d_RNSInvModqlqi; // Stores (ql)^-1 \pmod qi;

// fast_conv_B_to_Q and xi_ckks_rns_rid
__constant__ uint64_t d_RNSInvModBqi[MAX_COPRIMES_IN_A_BASE]; // Stores (B) \pmod qPi;

// rho_ckks_rns_rid
uint64_t *d_RNShatQQi; // Stores \hat{Qj} \pmod qi;

int round_up_blocks(int a, int b){
  return (a % b == 0? a / b: a / b + 1);
}

/**
 * @brief      Negates each coefficient of a
 *
 * @param      b          { parameter_description }
 * @param[in]  a          { parameter_description }
 * @param[in]  N          { parameter_description }
 * @param[in]  NResidues  The n residues
 */
__global__ void polynomial_negate(
  GaussianInteger *b,
  const GaussianInteger *a,
  const int N,
  const int NResidues ){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int size = N * NResidues;

  if(tid < size ){
    GaussianInteger x = a[tid];
    b[tid] = {
      x.re == 0? 
        0 :
        d_RNSCoprimes[rid] - (x.re),
      x.imag == 0? 
        0 :
        d_RNSCoprimes[rid] - (x.imag)
    };
  }

}

__host__ void CUDAEngine::execute_polynomial_negate(
    GaussianInteger *b,
    const GaussianInteger *a,
    const int base,
    Context *ctx ){
  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(base);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomial_negate<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    N,
    NResidues);
  cudaCheckError();
}

/**
 * @brief       Operate over each coefficient of a by an int x and write to b
 * 
 * @param[out] b         [description]
 * @param[in]  a         [description]
 * @param[in]  x         [description]
 * @param[in]  OP        [description]
 * @param[in]  N         [description]
 * @param[in]  NResidues [description]
 */
__global__ void polynomial_op_by_int(
  GaussianInteger *b,
  const GaussianInteger *a,
  uint64_t x,
  const add_mode_t OP,
  const int N,
  const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int size = N * NResidues;

  if(tid < size ){
    //  Coalesced access to global memory. Doing this way we reduce required
    // bandwich.
    //
    switch(OP){
    case ADD:
    b[tid] = GIAdd(a[tid], (GaussianInteger){x, 0}, rid);
    break;
    case SUB:
    b[tid] = GISub(a[tid], (GaussianInteger){x, 0}, rid);
    break;
    case MUL:
    b[tid].re   = mulmod(a[tid].re,   x, rid);
    b[tid].imag = mulmod(a[tid].imag, x, rid);
    break;
    default:
    printf("Unknown operation %d\n", OP);
    break;
    }
  }
}

__host__ void CUDAEngine::execute_polynomial_op_by_int(
    GaussianInteger *b,
    GaussianInteger *a,
    uint64_t x,
    const int base,
    add_mode_t OP,
    Context *ctx ){

  const int size = N * get_n_residues(base);
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomial_op_by_int<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    x,
    OP,
    N,
    get_n_residues(base));
  cudaCheckError();

}

/**
 * @brief       Operate over each coefficient of a by an int x and write to b
 * 
 * @param[out] b         [description]
 * @param[in]  a         [description]
 * @param[in]  x         [description]
 * @param[in]  OP        [description]
 * @param[in]  N         [description]
 * @param[in]  NResidues [description]
 */
__global__ void polynomial_double_op_by_int(
  GaussianInteger *b1,
  const GaussianInteger *a1,
  GaussianInteger *b2,
  const GaussianInteger *a2,
  uint64_t x1,
  uint64_t x2,
  const add_mode_t OP,
  const int N,
  const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int size = N * NResidues;

  if(tid < size ){
    //  Coalesced access to global memory. Doing this way we reduce required
    // bandwich.
    //
    switch(OP){
    case MULMUL:
      b1[tid].re = mulmod(a1[tid].re, x1, rid);
      b1[tid].imag = mulmod(a1[tid].imag, x1, rid);
      b2[tid].re = mulmod(a2[tid].re, x2, rid);
      b2[tid].imag = mulmod(a2[tid].imag, x2, rid);
    break;
    case ADDADD:
      b1[tid] = GIAdd(a1[tid], (GaussianInteger){x1, 0}, rid);
      b2[tid] = GIAdd(a2[tid], (GaussianInteger){x2, 0}, rid);
    break;
    default:
    printf("Unknown operation %d\n", OP);
    break;
    }
  }
}

__host__ void CUDAEngine::execute_polynomial_double_op_by_int(
    GaussianInteger *b1,
    GaussianInteger *a1,
    GaussianInteger *b2,
    GaussianInteger *a2,
    uint64_t x1,
    uint64_t x2,
    add_mode_t OP,
    const poly_bases base,
    Context *ctx ){

  const int size = N * get_n_residues(base);
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomial_double_op_by_int<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b1, a1,
    b2, a2,
    x1, x2,
    OP, N,
    get_n_residues(base));
  cudaCheckError();

}

/////////
//     //
// RNS //
//     //
/////////

/** Multiply a 128 bits uint64_t by a 64 bits uint64_t and returns the two most 
 significant words
 */
__device__ uint128_t __umul128hi(const uint128_t a, const uint64_t b){
  uint128_t c;
  c.lo = __umul64hi(a.lo, b);
  c.lo += a.hi * b;
  c.hi = __umul64hi(a.hi, b) + (c.lo < a.hi * b);

  return c;
}

__global__ void fast_conv_B_to_Q(
  GaussianInteger *b,
  GaussianInteger *a,
  const int N,
  const int level,
  uint64_t *RNSBbiqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_Q = tid / N; 
  GaussianInteger val = (GaussianInteger){0, 0};

  if(tid < N * (level + 1)){
    for(int rid_B = 0; rid_B < d_RNSBNPrimes; rid_B++){
      GaussianInteger x = a[cid + (rid_B + d_RNSQNPrimes) * N];

      GaussianInteger aux1 = 
          mulint_dgt(
            x, d_RNSInvModBbi[rid_B],
            rid_B + d_RNSQNPrimes);
      GaussianInteger aux2 = 
        mulint_dgt(aux1,
          RNSBbiqi[rid_B + rid_Q*d_RNSBNPrimes],
          rid_Q);
      val = GIAdd(
        val,
        aux2,
        rid_Q);
    }

    mulint_dgt(
      &b[cid + rid_Q * N], // Output
      GISub(
        a[cid + rid_Q * N],
        val,
        rid_Q),
      d_RNSInvModBqi[rid_Q],
      rid_Q);

  }
}

__host__ void CUDAEngine::execute_approx_modulus_reduction(
  Context *ctx,
  GaussianInteger *a,
  GaussianInteger *b,
  int level){

  const int N = CUDAEngine::N;
  const int NResiduesQ = level + 1;
  const int size = N * NResiduesQ;
  const int defaultblocksize = 32;
  const int ADDGRIDXDIM = (size%defaultblocksize == 0? size/defaultblocksize : size/defaultblocksize + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(defaultblocksize);

    fast_conv_B_to_Q<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      a, b, N,
      level,
      d_RNSBbiqi) ;
    cudaCheckError();

}


__global__ void ckks_rescale(
  GaussianInteger *a,  GaussianInteger *b,
  const int level,
  const int N,  const int nresidues_Q,
  uint64_t *RNSInvModqlqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid = tid / N; 

  if(tid < N * level){
    uint64_t s = RNSInvModqlqi[rid + level * d_RNSQNPrimes];
    __syncthreads();

    GaussianInteger *x = (threadIdx.y? b : a);

    GaussianInteger rescale = x[cid + level * N];
    __syncthreads();
    GaussianInteger coeff   = x[cid + rid * N];
    
    coeff.re   = submod(coeff.re,   rescale.re, rid);
    coeff.imag = submod(coeff.imag, rescale.imag, rid);

    GIMul(&coeff, coeff, s, rid);

    x[cid + rid * N] = coeff;

  }
}

__host__ void CUDAEngine::execute_ckks_rescale(
  GaussianInteger *a,
  GaussianInteger *b,
  const int level,
  Context *ctx){

  const int N = CUDAEngine::N;
  const int NResiduesQ = level + 1;
  const int size = N * NResiduesQ;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM, 2);

    ckks_rescale<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      a, b,
      level,
      N,
      NResiduesQ,
      d_RNSInvModqlqi);
    cudaCheckError();
}


__global__ void fast_conv_Q_to_B(
  GaussianInteger *a,
  const int N,
  const int level,
  uint64_t *RNSQqibi,
  uint64_t *RNSInvModQqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_B = tid / N; 
  GaussianInteger val = (GaussianInteger){0, 0};

  if(tid < N * d_RNSBNPrimes){
    for(int rid_Q = 0; rid_Q <= level; rid_Q++){
      GaussianInteger aux = 
          mulint_dgt(
            a[cid + rid_Q * N],
            RNSInvModQqi[rid_Q + level * d_RNSQNPrimes],
            rid_Q);
      aux.re %= d_RNSCoprimes[rid_B   + d_RNSQNPrimes];
      aux.imag %= d_RNSCoprimes[rid_B + d_RNSQNPrimes];
      val = GIAdd(
        val,
        mulint_dgt(aux,
          RNSQqibi[level * d_RNSQNPrimes * d_RNSBNPrimes + rid_Q * d_RNSBNPrimes + rid_B], //  [level][rid_Q][rid_B],
          rid_B + d_RNSQNPrimes),
        rid_B + d_RNSQNPrimes);
    }
    a[cid + (rid_B + d_RNSQNPrimes) * N] = val;
  }
}


__host__ void CUDAEngine::execute_approx_modulus_raising(
  Context *ctx,
  GaussianInteger *a,
  int level){

  const int N = CUDAEngine::N;
  const int NResiduesB = get_n_residues(BBase);
  const int size = N * NResiduesB;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

    fast_conv_Q_to_B<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      a,
      N,
      level,
      d_RNSQqibi,
      d_RNSInvModQqi) ;
    cudaCheckError();
}

/**
 * @brief     Apply \f$\rho()\f$ as in the CKKS.
 * 
 * @param[out] b         output
 * @param[in]  a         input
 * @param[in]  N         Degree of each residue
 * @param[in]  NResidues Quantity of residues
 */
__global__ void rho_ckks_rns_rid(
  GaussianInteger *b,
  const GaussianInteger *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int cid = tid % N;

  if(tid < N * NResidues)
      mulint_dgt(
        &b[cid + rid * N],
        a[cid + rid * N],
        d_RNSBqi[rid],
        rid
      );
}

__host__ void CUDAEngine::execute_rho_ckks_rns(
  GaussianInteger *b,
  GaussianInteger *a,
  Context *ctx){

  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(QBase);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);


  rho_ckks_rns_rid<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    N,
    NResidues);
  cudaCheckError();

}

/**
 * @brief     Apply \f$\xi()\f$ as in the CKKS.
 * 
 * @param[out] b         output
 * @param[in]  a         input
 * @param[in]  N         Degree of each residue
 * @param[in]  NResidues Quantity of residues
 */
__global__ void xi_ckks_rns_rid(
  GaussianInteger *b,
  const GaussianInteger *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int cid = tid % N;

  if(tid < N * NResidues)
      mulint_dgt(
        &b[cid + rid * N],
        a[cid + rid * N],
        d_RNSInvModBqi[rid],
        rid
        );
}

__host__ void CUDAEngine::execute_xi_ckks_rns(
  GaussianInteger *b,
  GaussianInteger *a,
  Context *ctx){

  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(QBase);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  xi_ckks_rns_rid<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    N,
    NResidues);
  cudaCheckError();

}

//////////////////////////
//                      //
//   Init / Destroy RNS //
//                      //
//////////////////////////

void fill_ckks_coprimes_bucket(int k, int kl, int prec){
  switch(prec){

    case 45:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    assert(COPRIMES_45_BUCKET_SIZE >= (k + kl - 1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_45_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_45_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    case 48:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    assert(COPRIMES_55_BUCKET_SIZE >= (k + kl -1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_48_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    case 52:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    assert(COPRIMES_55_BUCKET_SIZE >= (k + kl -1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_52_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    case 55:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    // COPRIMES_BUCKET_SIZE = COPRIMES_55_BUCKET_SIZE + 1;
    assert(COPRIMES_55_BUCKET_SIZE >= (k + kl -1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_55_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    default:
    throw std::runtime_error("AOADGT can't handle this precision.");
  }
  COPRIMES_BUCKET_SIZE = (k+kl);
}

// Multiply all elements of q with index j such that a <= j < b
template <class T>
ZZ multiply_subset(std::vector<T> q, int a, int b){
  ZZ accum = to_ZZ(1);
  for(int i = a; i < b; i++)
    accum *= to_ZZ(q[i]);
  return accum;
}

// Multiply all elements of q with index j such that a <= j < b and j != c
template <class T>
ZZ multiply_subset_except(std::vector<T> q, int a, int b, int c){
  ZZ accum = multiply_subset(q, a, b);
  return accum / q[c];
}

// Selects a set of coprimes to be used as the main and secondary bases
__host__ void CUDAEngine::gen_rns_primes(
  unsigned int k,       // length of |q|
  unsigned int kl){     // length of |b|  

  // Loads to COPRIMES_BUCKET all the coprimes that may be used 
  fill_ckks_coprimes_bucket(k, kl, scalingfactor);

  // Select the main base
  RNSProduct = to_ZZ(1);
  ostringstream os;
  os << "Q base: ";
  for(unsigned int i = 0; RNSPrimes.size() < k; i++){
    assert(
      RNSPrimes.size() < MAX_COPRIMES_IN_A_BASE &&
      RNSPrimes.size() < COPRIMES_BUCKET_SIZE);
    RNSPrimes.push_back(COPRIMES_BUCKET[RNSPrimes.size()]);
    assert(RNSPrimes.back() > 0);
    RNSProduct *= RNSPrimes.back();
    
    os << RNSPrimes.back() << " ";
  }
  os << std::endl;

  // Select the secondary base
  RNSBProduct = to_ZZ(1);
  os << "B base: ";
  for(unsigned int i = 0; i < kl; i++){ // |B| == |q| + 1 
    assert(
      RNSBPrimes.size() < MAX_COPRIMES_IN_A_BASE &&
      RNSPrimes.size() + RNSBPrimes.size() < COPRIMES_BUCKET_SIZE);
    RNSBPrimes.push_back(COPRIMES_BUCKET[RNSPrimes.size() + RNSBPrimes.size()]);
    assert(RNSBPrimes.back() > 0);
    RNSBProduct *= RNSBPrimes.back();

    os << RNSBPrimes.back() << " ";
  }
  os << std::endl;
  
  // Copy to device
  assert(RNSPrimes.size() < MAX_COPRIMES_IN_A_BASE && RNSPrimes.size() > 0);
  assert(RNSBPrimes.size() < MAX_COPRIMES_IN_A_BASE && RNSBPrimes.size() > 0);
  assert(RNSPrimes.size() + RNSBPrimes.size() + 1 < MAX_COPRIMES);
  
  cudaMemcpyToSymbol(
    d_RNSCoprimes,
    &RNSPrimes[0],
    RNSPrimes.size() * sizeof(uint64_t),
    0);
  cudaCheckError();

  cudaMemcpyToSymbol(
    d_RNSCoprimes,
    &RNSBPrimes[0],
    RNSBPrimes.size() * sizeof(uint64_t),
    RNSPrimes.size() * sizeof(uint64_t)); // offset
  cudaCheckError();

  int vsize = RNSPrimes.size(); // Do we need this
  cudaMemcpyToSymbol(
    d_RNSQNPrimes,
    &vsize,
    sizeof(int));
    cudaCheckError();
    
    vsize = RNSBPrimes.size(); // Do we need this
    cudaMemcpyToSymbol(
      d_RNSBNPrimes,
      &vsize,
    sizeof(int));
  cudaCheckError();

  ///////////////////////
  cudaDeviceSynchronize();
  cudaCheckError();

  os << "q: " << RNSProduct << " ( " << NumBits(RNSProduct) << " bits )" << std::endl;
  os << "B: " << RNSBProduct << " ( " << NumBits(RNSBProduct) << " bits )" << std::endl;
  os << "|q| == " << RNSPrimes.size() << std::endl;
  os << "|B| == " << RNSBPrimes.size() << std::endl;
  Logger::getInstance()->log_info(os.str().c_str());

}

__host__ void CUDAEngine::precompute(){
  const int k = RNSPrimes.size();
  const int kl = RNSBPrimes.size();
  
  std::vector<uint64_t> Qqi;
  std::vector<uint64_t> Qbi;
  std::vector<uint64_t> Bqi;
  std::vector<uint64_t> Qqibi;
  std::vector<uint64_t> InvBbi;
  std::vector<uint64_t> Bbiqi;
  std::vector<uint64_t> InvModBqi;

  std::vector<uint64_t> hatQQi;
  uint64_t Invqlqi [MAX_COPRIMES_IN_A_BASE][MAX_COPRIMES_IN_A_BASE];
  uint64_t InvQqi  [MAX_COPRIMES_IN_A_BASE][MAX_COPRIMES_IN_A_BASE];

  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // Compute Q/qi mod qi and it's inverse
  for(auto p : RNSPrimes){
    ZZ pi = to_ZZ(p);

    RNSMpi.push_back(RNSProduct/pi);
    
    // rho_bfv_rns_rid
    Qqi.push_back(conv<uint64_t>(RNSMpi.back() % pi));
    
    // HPS basis extension  and fast_conv_Q_to_B
    RNSInvMpi.push_back(conv<uint64_t>(NTL::InvMod(to_ZZ(Qqi.back()), pi)));

    //  B % qi
    // rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
    Bqi.push_back(conv<uint64_t>(RNSBProduct % pi));

    //  (B % qi)^-1 mod qi
    // fast_conv_B_to_Q and xi_ckks_rns_rid
    InvModBqi.push_back(
      conv<uint64_t>(NTL::InvMod(RNSBProduct % pi, pi))
    );
  }

  for(auto p : RNSBPrimes){
    ZZ bi = to_ZZ(p);

    // Compute Q mod bi
    Qbi.push_back(conv<uint64_t>(RNSProduct % bi));
  
    // Compute B/bi and it's inverse
    InvBbi.push_back(conv<uint64_t>(NTL::InvMod(RNSBProduct / bi % bi, bi)));
  }

  // HPS basis extension from B to Q and fast_conv_B_to_Q
  for(auto qi : RNSPrimes)
    for(auto bi : RNSBPrimes)
      Bbiqi.push_back(conv<uint64_t>(RNSBProduct / to_ZZ(bi) % to_ZZ(qi)));    

  
    for(int l = 0; l < k; l++){ // Level
      for(int i = 0; i < k; i++){ // Q residue
        ZZ qi = to_ZZ(RNSPrimes[i]);
        if( i < k )
          InvQqi[l][i] = conv<uint64_t>(
            InvMod(
              multiply_subset_except(RNSPrimes, 0, l+1, i) % qi,
              qi));
        for(int j = 0; j < kl; j++){ // B residue
          ZZ pi = to_ZZ(RNSBPrimes[j]);
  
          Qqibi.push_back(
            conv<uint64_t>(
              multiply_subset_except(RNSPrimes, 0, l+1, i) % pi)
            );
        }
      }
    }

  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // CKKS' 
    // alpha = (L + 1) /dnum
    int alpha = k / CUDAEngine::dnum;
    std::vector<ZZ> Qj(CUDAEngine::dnum, to_ZZ(1));
    for(int j = 0; j < CUDAEngine::dnum; j++)
      for(int i = j * alpha; i < (j + 1) * alpha; i++)
        Qj[j] *= RNSPrimes[i];

    std::vector<ZZ> hatQj(CUDAEngine::dnum, to_ZZ(1));
    for(int j = 0; j < CUDAEngine::dnum; j++)
      for(int i = 0; i < CUDAEngine::dnum; i++)
        if( i != j)
          hatQj[j] *= Qj[i];

    for(int j = 0; j < CUDAEngine::dnum; j++)
      for(int i = 0; i < k; i++)
        hatQQi.push_back(conv<uint64_t>(hatQj[j] % RNSPrimes[i]));

    // Compute ql^{-1} mod qi
    for(unsigned int l = 0; l < RNSPrimes.size();l++)
      for(unsigned int i = 0; i < RNSPrimes.size();i++)
        if(l != i)
          Invqlqi[l][i] = conv<uint64_t>(
            NTL::InvMod(to_ZZ(RNSPrimes[l]) % to_ZZ(RNSPrimes[i]),
            to_ZZ(RNSPrimes[i])));
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  RR::SetPrecision(512);

  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // Linearizes InvQqi
  uint64_t *lin_InvQqi = (uint64_t*) malloc (k * k * sizeof(uint64_t));
  uint64_t *lin_Invqlqi = (uint64_t*) malloc (k * k * sizeof(uint64_t));

  // Copy to device
  // CKKS rescale
  cudaMalloc((void**)&d_RNSInvModqlqi, k * k * sizeof(uint64_t));
  cudaCheckError();
  for(int i = 0; i < k; i++)
    for(int j = 0; j < k; j++)
      lin_Invqlqi[j + i * k] = Invqlqi[i][j];
  cudaMemcpyAsync(
    d_RNSInvModqlqi,
    lin_Invqlqi,
    k * k * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  // HPS basis extension  and fast_conv_Q_to_B
  cudaMalloc((void**)&d_RNSInvModQqi, Qqibi.size() * sizeof(uint64_t));
  cudaCheckError();
  for(int i = 0; i < k; i++)
    for(int j = 0; j < k; j++)
      lin_InvQqi[j + i * k] = InvQqi[i][j];
  cudaMemcpyAsync(
    d_RNSInvModQqi,
    lin_InvQqi,
    k * k * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  assert(Qqibi.size() < MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE);
  cudaMalloc((void**)&d_RNSQqibi, Qqibi.size() * sizeof(uint64_t));
  cudaCheckError();

  // HPS basis extension Q to B and fast_conv_Q_to_B
  cudaMemcpyAsync(
    d_RNSQqibi,
    &Qqibi[0],
    Qqibi.size() * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  assert(InvModBqi.size() < MAX_COPRIMES_IN_A_BASE);
  // fast_conv_B_to_Q and xi_ckks_rns_rid
  cudaMemcpyToSymbol(
    d_RNSInvModBqi,
    &InvModBqi[0],
    InvModBqi.size() * sizeof(uint64_t));
  cudaCheckError();

  assert(Bbiqi.size() < MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE);
  // HPS basis extension from B to Q and fast_conv_B_to_Q
  cudaMalloc((void**)& d_RNSBbiqi, Bbiqi.size() * sizeof(uint64_t));
  cudaCheckError();
  cudaMemcpyAsync(
    d_RNSBbiqi,
    &Bbiqi[0],
    Bbiqi.size() * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  // rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
  assert(Bqi.size() < MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSBqi,
    &Bqi[0],
    Bqi.size() * sizeof(uint64_t));
  cudaCheckError();

  assert(InvBbi.size() < MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSInvModBbi,
    &InvBbi[0],
    InvBbi.size()*sizeof(uint64_t));
  cudaCheckError();

  cudaDeviceSynchronize();
  cudaCheckError();
  free(lin_InvQqi);
}

//////////////////////
//                  //
//   Init / Destroy //
//                  //
//////////////////////

__host__ void CUDAEngine::init(CUDAParams p){
  

    init(p.k, p.kl, p.nphi, p.pt);
}

__host__ bool is_power2(uint64_t t){
  return t && !(t & (t - 1));
}

__host__ void CUDAEngine::init(
  const int k,
  const int kl,
  const int M,
  const uint64_t t){
  ostringstream os;

  // By using the folded encoding for DGT we just need half the degree
  CUDAEngine::N = (M >> 1);

  ////////////////////////
  // Generate RNSPrimes //
  ////////////////////////    
  scalingfactor = t;
  dnum = 1;
  gen_rns_primes(k, (kl > 0? kl : k+1));// Generates CRT's primes
  precompute();
  cudaGetSymbolAddress((void**)&CUDAEngine::RNSCoprimes, d_RNSCoprimes);
  cudaCheckError();
  is_init = true;

  ///////////////
  // Greetings //
  ///////////////
  os << "CUDAEngine initialized  = [" << CUDAEngine::N << "]" << std::endl;
  Logger::getInstance()->log_info(os.str().c_str());  
  
  /////////
  // DGT //
  /////////
  DGTEngine::init();    
}

__host__ void CUDAEngine::destroy(){
  if(!is_init)
    return;

  CUDAEngine::RNSPrimes.clear();
  CUDAEngine::RNSMpi.clear();
  CUDAEngine::RNSInvMpi.clear();
  CUDAEngine::RNSBPrimes.clear();

  cudaFree(d_RNSBbiqi);
  cudaCheckError();
  cudaFree(d_RNSInvModQqi);
  cudaCheckError();
  cudaFree(d_RNSQqibi);
  cudaCheckError();
  cudaFree(d_RNShatQQi);
  cudaCheckError();
  
  DGTEngine::destroy();

  cudaDeviceSynchronize();
  cudaCheckError();

  is_init = false;
}

/**
 * Return the quantity of residues for a certain base
 */
__host__ int CUDAEngine::get_n_residues(int base){
  switch(base){
    case QBase:
      return RNSPrimes.size();
    case TBase:
      return 1;
    case BBase:
      return RNSBPrimes.size();
    case QBBase:
      return RNSPrimes.size() + RNSBPrimes.size();
    case QTBBase:
      return RNSPrimes.size() + RNSBPrimes.size() + 1;
    default:
      throw std::runtime_error("Unknown base!");
  }
}
