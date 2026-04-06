#include <sstream>
#include <newckks/cuda/manager.h>
#include <newckks/coprimes.h>

using namespace NTL;

std::vector<uint64_t> CUDAManager::RNSQPrimes;
std::vector<uint64_t> CUDAManager::RNSBPrimes;
int CUDAManager::N = 0;
int CUDAManager::is_init = false;
int CUDAManager::dnum = 1;
uint64_t CUDAManager::scalingfactor = 1;

// \todo These shall be moved to inside CUDAManager
uint32_t COPRIMES_BUCKET_SIZE; //!< The size of COPRIMES_55_BUCKET.

///////////////
// Main base //
///////////////
__constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];
__constant__ int      d_RNSQNPrimes;// Used primes

// HPS basis extension from B to Q and fast_conv_B_to_Q
__constant__ int      d_RNSBNPrimes;// Used primes
                                    // 
// rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
__constant__ uint64_t d_RNSBqi[MAX_COPRIMES_IN_A_BASE]; // Stores (B) \pmod qPi;
                                                        // 
// CKKS rescale
uint64_t *d_RNSInvModqlqi; // Stores (ql)^-1 \pmod qi;

// fast_conv_B_to_Q
uint64_t *d_RNSBbiqi; // Stores (B/bi) \pmod qi;
                      // 
// HPS basis extension Q to B and fast_conv_Q_to_B
uint64_t *d_RNSQqibi; // Stores (Q/qi) \pmod bi; ; for each level

// fast_conv_Q_to_B
uint64_t *d_RNSInvModQqi; // Stores (Q/bi)^-1 \pmod qi; for each level

// fast_conv_B_to_Q
__constant__ uint64_t d_RNSInvModBbi[MAX_COPRIMES_IN_A_BASE]; // Stores (B/bi)^-1 \pmod bi;

// fast_conv_B_to_Q and xi_ckks_rns_rid
__constant__ uint64_t d_RNSInvModBqi[MAX_COPRIMES_IN_A_BASE]; // Stores (B) \pmod qPi;
//////////////////////
// Basic operations //
//////////////////////

__global__ void dot_prod(
  uint64_t *c,
  uint64_t **a,
  uint64_t **b,
  const int k,
  const int N, const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  uint128_t v = 0;

  if(tid < N * NResidues){
    for(int i = 0; i < k; i++)
      v = add128(v, mullazy(a[i][tid], b[i][tid]), rid);
    c[tid] = mod(v, rid);
  }
}

void CUDAManager::execute_dot(
  Context *ctx,
  uint64_t *c,
  uint64_t **a,
  uint64_t **b,
  const int k){

  const int size = N * CUDAManager::get_n_residues(QBBase);
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  dot_prod<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    c, a, b, k,
    N, CUDAManager::get_n_residues(QBBase));
  cudaCheckError();

}

//////////
// CKKS //
//////////
__global__ void ckks_rescale(
  uint64_t *a,
  uint64_t *b,
  const int level,
  const int N,
  const int nresidues_Q,
  uint64_t *RNSInvModqlqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid = tid / N; 

  if(tid < N * level){
    uint64_t s = RNSInvModqlqi[rid + level * d_RNSQNPrimes];
    uint64_t coeff_a = a[cid + rid * N];
    uint64_t rescale_a = a[cid + level * N];
    uint64_t coeff_b = b[cid + rid * N];
    uint64_t rescale_b = b[cid + level * N];

    coeff_a = submod(coeff_a, rescale_a, rid),
    coeff_b = submod(coeff_b, rescale_b, rid),

    coeff_a = mulmod(coeff_a, s, rid);
    coeff_b = mulmod(coeff_b, s, rid);

    a[cid + rid * N] = coeff_a;
    b[cid + rid * N] = coeff_b;
  }
}

void CUDAManager::execute_rescale(
  Context *ctx,
  uint64_t *a,
  uint64_t *b,
  const int level){

  const int N = CUDAManager::N;
  const int NResiduesQ = level + 1;
  const int size = N * NResiduesQ;
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  ckks_rescale<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    a, b,
    level,
    N, NResiduesQ,
    d_RNSInvModqlqi);
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
__global__ void ckks_rho(
  uint64_t *b,
  const uint64_t *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;

  if(tid < N * NResidues)
      b[tid] = mulmod(a[tid], d_RNSBqi[rid], rid);
}

__host__ void CUDAManager::execute_rho(
  Context *ctx,
  uint64_t *b,
  uint64_t *a){

  const int size = CUDAManager::N * CUDAManager::get_n_residues(QBBase);
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  ckks_rho<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b, a,
    CUDAManager::N, CUDAManager::get_n_residues(QBBase));
  cudaCheckError();

}

#ifdef OPTMODUP

__global__ void fast_conv_Q_to_B_opt(
  uint64_t *b,
  uint64_t *a,
  const int N,
  const int level,
  uint64_t *RNSQqibi,
  uint64_t *RNSInvModQqi){

  const int Nh = (N>>1);
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % Nh; 
  const int rid_B = tid / Nh; 
  uint64_t val1 = 0;
  uint64_t val2 = 0;

  if(tid < Nh * d_RNSBNPrimes){
    for(int rid_Q = 0; rid_Q <= level; rid_Q++){
      uint64_t aux1 = 
          mulmod(
            a[cid + rid_Q * N],
            RNSInvModQqi[rid_Q + level * d_RNSQNPrimes],
            rid_Q);
      uint64_t aux2 = 
          mulmod(
            a[cid + Nh + rid_Q * N],
            RNSInvModQqi[rid_Q + level * d_RNSQNPrimes],
            rid_Q);
      aux1 %= d_RNSCoprimes[rid_B   + d_RNSQNPrimes];
      aux2 %= d_RNSCoprimes[rid_B   + d_RNSQNPrimes];

      uint64_t c = RNSQqibi[
        level * d_RNSQNPrimes * d_RNSBNPrimes +
        rid_Q * d_RNSBNPrimes +
        rid_B]; //  [level][rid_Q][rid_B],
      
      uint64_t tmp1 = mulmod(aux1,c, rid_B + d_RNSQNPrimes);
      uint64_t tmp2 = mulmod(aux2,c, rid_B + d_RNSQNPrimes);
      val1 = addmod( val1, tmp1, rid_B + d_RNSQNPrimes);
      val2 = addmod( val2, tmp2, rid_B + d_RNSQNPrimes);
    }
    b[cid + (rid_B + d_RNSQNPrimes) * N] = val1;
    b[cid + Nh + (rid_B + d_RNSQNPrimes) * N] = val2;
  }
}

void CUDAManager::execute_modup(
  Context *ctx,
  uint64_t *b,
  uint64_t *a,
  int level){

  const int NResiduesB = CUDAManager::RNSBPrimes.size();
  const int size = (CUDAManager::N>>1) * NResiduesB;
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  switch(COMMONEngine::engine_mode){
    case NTTTrans:
    
    fast_conv_Q_to_B_opt<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      b, a, 
      CUDAManager::N, level, 
      d_RNSQqibi, d_RNSInvModQqi) ;
    cudaCheckError()
    
    break;
    default:
    throw std::runtime_error("Unknown engine mode.");
  }
}

#else


__global__ void fast_conv_Q_to_B_nonopt(
  uint64_t *b,
  uint64_t *a,
  const int N,
  const int level,
  uint64_t *RNSQqibi,
  uint64_t *RNSInvModQqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_B = tid / N; 
  uint64_t val = 0;

  if(tid < N * d_RNSBNPrimes){
    for(int rid_Q = 0; rid_Q <= level; rid_Q++){
      uint64_t aux = 
          mulmod(
            a[cid + rid_Q * N],
            RNSInvModQqi[rid_Q + level * d_RNSQNPrimes],
            rid_Q);
      aux %= d_RNSCoprimes[rid_B   + d_RNSQNPrimes];
      val = addmod(
        val,
        mulmod(aux,
          RNSQqibi[level * d_RNSQNPrimes * d_RNSBNPrimes + rid_Q * d_RNSBNPrimes + rid_B], //  [level][rid_Q][rid_B],
          rid_B + d_RNSQNPrimes),
        rid_B + d_RNSQNPrimes);
    }
    b[cid + (rid_B + d_RNSQNPrimes) * N] = val;
  }
}

void CUDAManager::execute_modup(
  Context *ctx,
  uint64_t *b,
  uint64_t *a,
  int level){

  const int NResiduesB = CUDAManager::RNSBPrimes.size();
  const int size = CUDAManager::N * NResiduesB;
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  switch(COMMONEngine::engine_mode){
    case NTTTrans:
    
    fast_conv_Q_to_B_nonopt<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      b, a, 
      CUDAManager::N, level, 
      d_RNSQqibi, d_RNSInvModQqi) ;
    cudaCheckError()
    
    break;
    default:
    throw std::runtime_error("Unknown engine mode.");
  }
}
#endif
__global__ void fast_conv_B_to_Q(
  uint64_t *b,
  uint64_t *a,
  const int N,
  const int level,
  uint64_t *RNSBbiqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_Q = tid / N; 
  uint64_t val1 = 0;

  if(tid < N * (level + 1)){
    for(int rid_B = 0; rid_B < d_RNSBNPrimes; rid_B++){
      uint64_t aux1 = 
          mulmod(
            a[cid + (rid_B + d_RNSQNPrimes) * N],
            d_RNSInvModBbi[rid_B],
            rid_B + d_RNSQNPrimes);
      uint64_t tmp = 
        mulmod(aux1,
          RNSBbiqi[rid_B + rid_Q*d_RNSBNPrimes],
          rid_Q);
      val1 = addmod(val1, tmp, rid_Q);
    }

    uint64_t tmp = submod(a[cid + rid_Q * N], val1, rid_Q);
    b[cid + rid_Q * N] = mulmod(tmp, d_RNSInvModBqi[rid_Q], rid_Q);
  }
}


void CUDAManager::execute_moddown(
  Context *ctx,
  uint64_t *b,
  uint64_t *a,
  int level){

  const int NResiduesQ = level + 1;
  const int size = CUDAManager::N * NResiduesQ;
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  switch(COMMONEngine::engine_mode){
    case NTTTrans:
    fast_conv_B_to_Q<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      b, a,
      CUDAManager::N, level,
      d_RNSBbiqi) ;
    cudaCheckError();
    break;
    default:
    throw std::runtime_error("Unknown engine mode.");
  }

}

__global__ void dr2(
  uint64_t *ct21, // Outcome
  uint64_t *ct22, // Outcome
  uint64_t *ct23, // Outcome
  const uint64_t *ct01, // Operand 1
  const uint64_t *ct02, // Operand 1
  const uint64_t *ct11, // Operand 2
  const uint64_t *ct12, // Operand 2
  const int N,
  const int nresidues){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int rid = tid / N;

  if(tid < N * nresidues){
    uint64_t local_ct01 = ct01[tid], local_ct02 = ct02[tid];
    uint64_t local_ct11 = ct11[tid], local_ct12 = ct12[tid];

    ct21[tid] = mulmod(local_ct01, local_ct11, rid);

    ct22[tid] = mulmod(
      mulmod(local_ct01, local_ct12, rid),
      mulmod(local_ct02, local_ct11, rid),
      rid);

    ct23[tid] = mulmod(local_ct02, local_ct12, rid);
  }
}

__host__ void CUDAManager::execute_dr2(
  Context *ctx,
  uint64_t *ct21, // Outcome
  uint64_t *ct22, // Outcome
  uint64_t *ct23, // Outcome
  const uint64_t *ct01, // Operand 1
  const uint64_t *ct02, // Operand 1
  const uint64_t *ct11, // Operand 2
  const uint64_t *ct12){ // Operand 2


  const int size = CUDAManager::N * CUDAManager::get_n_residues(QBBase);
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  dr2<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
    ct21, ct22, ct23,
    ct01, ct02,
    ct11, ct12,
    N, CUDAManager::get_n_residues(QBBase));
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
    assert(COPRIMES_48_BUCKET_SIZE >= (k-1));
    assert(COPRIMES_55_BUCKET_SIZE >= kl);
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_48_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[0], kl    * sizeof(uint64_t)); // B Base
    break;

    case 52:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    assert(COPRIMES_52_BUCKET_SIZE >= (k-1));
    assert(COPRIMES_55_BUCKET_SIZE >= kl);
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_52_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[0], kl    * sizeof(uint64_t)); // B Base
    break;

    case 55:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    // COPRIMES_BUCKET_SIZE = COPRIMES_55_BUCKET_SIZE + 1;
    assert(COPRIMES_55_BUCKET_SIZE >= (k + kl -1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_55_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    default:
    throw std::runtime_error("We don't support this precision.");
  }
  COPRIMES_BUCKET_SIZE = (k+kl);
}

// Multiply all elements of q with index j such that a <= j < b
template <class T>
ZZ multiply_subset(std::vector<T> q, int a, int b){
  ZZ accum = NTL::to_ZZ(1);
  for(int i = a; i < b; i++)
    accum *= NTL::to_ZZ(q[i]);
  return accum;
}

// Multiply all elements of q with index j such that a <= j < b and j != c
template <class T>
ZZ multiply_subset_except(std::vector<T> q, int a, int b, int c){
  ZZ accum = multiply_subset(q, a, b);
  return accum / q[c];
}

// Selects a set of coprimes to be used as the main and secondary bases
void CUDAManager::gen_rns_primes(
  unsigned int k,       // length of |q|
  unsigned int kl){     // length of |b|  

  // Loads to COPRIMES_BUCKET all the coprimes that may be used 
  fill_ckks_coprimes_bucket(k, kl, scalingfactor);

  // Select the main base
  std::ostringstream os;
  os << "Q base: ";
  for(unsigned int i = 0; RNSQPrimes.size() < k; i++){
    assert(
      RNSQPrimes.size() < MAX_COPRIMES_IN_A_BASE &&
      RNSQPrimes.size() < COPRIMES_BUCKET_SIZE);
    RNSQPrimes.push_back(COPRIMES_BUCKET[RNSQPrimes.size()]);
    assert(RNSQPrimes.back() > 0);
    
    os << RNSQPrimes.back() << " ";
  }
  os << std::endl;

  // Select the secondary base
  os << "B base: ";
  for(unsigned int i = 0; i < kl; i++){ // |B| == |q| + 1 
    assert(
      RNSBPrimes.size() < MAX_COPRIMES_IN_A_BASE &&
      RNSQPrimes.size() + RNSBPrimes.size() < COPRIMES_BUCKET_SIZE);
    RNSBPrimes.push_back(COPRIMES_BUCKET[RNSQPrimes.size() + RNSBPrimes.size()]);
    assert(RNSBPrimes.back() > 0);

    os << RNSBPrimes.back() << " ";
  }
  os << std::endl;
  
  // Copy to device
  assert(RNSQPrimes.size() < MAX_COPRIMES_IN_A_BASE && RNSQPrimes.size() > 0);
  assert(RNSBPrimes.size() < MAX_COPRIMES_IN_A_BASE && RNSBPrimes.size() > 0);
  assert(RNSQPrimes.size() + RNSBPrimes.size() + 1 < MAX_COPRIMES);
  
  cudaMemcpyToSymbol(
    d_RNSCoprimes,
    &RNSQPrimes[0],
    RNSQPrimes.size() * sizeof(uint64_t),
    0);
  cudaCheckError();

  cudaMemcpyToSymbol(
    d_RNSCoprimes,
    &RNSBPrimes[0],
    RNSBPrimes.size() * sizeof(uint64_t),
    RNSQPrimes.size() * sizeof(uint64_t)); // offset
  cudaCheckError();

  int vsize = RNSQPrimes.size(); // Do we need this
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

  ZZ RNSQProduct = get_q_product();
  ZZ RNSBProduct = get_b_product();
  os << "q: " << RNSQProduct << " ( " << NumBits(RNSQProduct) << " bits )" << std::endl;
  os << "B: " << RNSBProduct << " ( " << NumBits(RNSBProduct) << " bits )" << std::endl;
  os << "|q| == " << RNSQPrimes.size() << std::endl;
  os << "|B| == " << RNSBPrimes.size() << std::endl;
  #ifdef ENABLE_DEBUG_MACRO
  std::cout << os.str() <<std::endl;
  #endif

}

void CUDAManager::precompute(){
  const int k = RNSQPrimes.size();
  const int kl = RNSBPrimes.size();

  std::vector<uint64_t> hatQQi;
  uint64_t Invqlqi [MAX_COPRIMES_IN_A_BASE][MAX_COPRIMES_IN_A_BASE];
  uint64_t InvQqi  [MAX_COPRIMES_IN_A_BASE][MAX_COPRIMES_IN_A_BASE];
  std::vector<uint64_t> Qqi;
  std::vector<uint64_t> Qbi;
  std::vector<uint64_t> Bqi;
  std::vector<uint64_t> Qqibi;
  std::vector<uint64_t> Bbiqi;
  std::vector<ZZ> RNSMpi;
  std::vector<uint64_t> RNSInvMpi;
  std::vector<uint64_t> InvModBqi;
  std::vector<uint64_t> InvBbi;

  ZZ RNSQProduct = get_q_product();
  ZZ RNSBProduct = get_b_product();

  for(auto p : RNSQPrimes){
    ZZ pi = to_ZZ(p);

    RNSMpi.push_back(RNSQProduct/pi);
    
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
    Qbi.push_back(conv<uint64_t>(RNSQProduct % bi));
  
    // Compute B/bi and it's inverse
    InvBbi.push_back(conv<uint64_t>(NTL::InvMod(RNSBProduct / bi % bi, bi)));
  }

  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // alpha = (L + 1) /dnum
  int alpha = k / CUDAManager::dnum;
  std::vector<ZZ> Qj(CUDAManager::dnum, NTL::to_ZZ(1));
  for(int j = 0; j < CUDAManager::dnum; j++)
    for(int i = j * alpha; i < (j + 1) * alpha; i++)
      Qj[j] *= RNSQPrimes[i];

  std::vector<ZZ> hatQj(CUDAManager::dnum, NTL::to_ZZ(1));
  for(int j = 0; j < CUDAManager::dnum; j++)
    for(int i = 0; i < CUDAManager::dnum; i++)
      if( i != j)
        hatQj[j] *= Qj[i];

  for(int j = 0; j < CUDAManager::dnum; j++)
    for(int i = 0; i < k; i++)
      hatQQi.push_back(NTL::conv<uint64_t>(hatQj[j] % RNSQPrimes[i]));

  // Compute ql^{-1} mod qi
  for(unsigned int l = 0; l < RNSQPrimes.size();l++)
    for(unsigned int i = 0; i < RNSQPrimes.size();i++)
      if(l != i)
        Invqlqi[l][i] = NTL::conv<uint64_t>(
          NTL::InvMod(NTL::to_ZZ(RNSQPrimes[l]) % NTL::to_ZZ(RNSQPrimes[i]),
          NTL::to_ZZ(RNSQPrimes[i])));

  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // HPS basis extension from B to Q and fast_conv_B_to_Q
  for(auto qi : RNSQPrimes)
    for(auto bi : RNSBPrimes)
      Bbiqi.push_back(conv<uint64_t>(RNSBProduct / to_ZZ(bi) % to_ZZ(qi))); 

  for(int l = 0; l < k; l++){ // Level
    for(int i = 0; i < k; i++){ // Q residue
      ZZ qi = to_ZZ(RNSQPrimes[i]);
      if( i < k )
        InvQqi[l][i] = conv<uint64_t>(
          InvMod(
            multiply_subset_except(RNSQPrimes, 0, l+1, i) % qi,
            qi));
      for(int j = 0; j < kl; j++){ // B residue
        ZZ pi = to_ZZ(RNSBPrimes[j]);

        Qqibi.push_back(
          conv<uint64_t>(
            multiply_subset_except(RNSQPrimes, 0, l+1, i) % pi)
          );
      }
    }
  }


  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 

  // Linearizes InvQqi
  uint64_t *lin_InvQqi = (uint64_t*) malloc (k * k * sizeof(uint64_t));
  uint64_t *lin_Invqlqi = (uint64_t*) malloc (k * k * sizeof(uint64_t));

  // fast_conv_Q_to_B
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

  cudaDeviceSynchronize();
  cudaCheckError();
  free(lin_InvQqi);
  free(lin_Invqlqi);

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

  assert(Qqibi.size() < MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE);
  cudaMalloc((void**)&d_RNSQqibi, Qqibi.size() * sizeof(uint64_t));
  cudaCheckError();

  // fast_conv_Q_to_B
  cudaMemcpyAsync(
    d_RNSQqibi,
    &Qqibi[0],
    Qqibi.size() * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  assert(InvBbi.size() < MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSInvModBbi,
    &InvBbi[0],
    InvBbi.size()*sizeof(uint64_t));
  cudaCheckError();

  assert(InvModBqi.size() < MAX_COPRIMES_IN_A_BASE);
  // fast_conv_B_to_Q and xi_ckks_rns_rid
  cudaMemcpyToSymbol(
    d_RNSInvModBqi,
    &InvModBqi[0],
    InvModBqi.size() * sizeof(uint64_t));
  cudaCheckError();

  // rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
  assert(Bqi.size() < MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSBqi,
    &Bqi[0],
    Bqi.size() * sizeof(uint64_t));
  cudaCheckError();
}


void CUDAManager::init(
  const int k,
  const int kl,
  const int N,
  const uint64_t scalingfactor,
  engine_types e){

  if(e != NTTTrans)
    throw std::runtime_error("Unknown engine mode");
    
  CUDAManager::N = N;

  ////////////////////////
  // Generate RNSQPrimes //
  ////////////////////////    
  CUDAManager::dnum = 1;
  CUDAManager::scalingfactor = scalingfactor;
  gen_rns_primes(k, (kl > 0? kl : k+1));// Generates CRT's primes
  precompute();

  assert(CUDAManager::N > 0);
  assert(CUDAManager::get_n_residues(QBBase) > 0);
  is_init = true;

  ///////////////
  // Greetings //
  ///////////////
  #ifdef NDEBUG
  std::cout << "CUDAManager initialized  = [" << CUDAManager::N << "]" << std::endl;
  #endif
  /////////
  // NTT //
  /////////
  COMMONEngine::init(e);    

  /////////
  cudaDeviceSynchronize();
  cudaCheckError();   
}

void CUDAManager::destroy(){
  if(!is_init)
    return;

  cudaDeviceSynchronize();
  cudaCheckError();

  CUDAManager::RNSQPrimes.clear();
  CUDAManager::RNSBPrimes.clear();

  cudaFree(d_RNSInvModqlqi);
  cudaCheckError();
  cudaFree(d_RNSBbiqi);
  cudaCheckError();
  cudaFree(d_RNSInvModQqi);
  cudaCheckError();
  
  COMMONEngine::destroy();

  is_init = false;
}