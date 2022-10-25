#include <newckks/cuda/htrans/ntt.h>
#include <newckks/ckks/ckkscontext.h>
#include <newckks/coprimes.h>
#include <omp.h>

using namespace NTL;

bool NTTEngine::is_init = false;
std::map<int, uint64_t*> NTTEngine::d_nthroot;
std::map<int, uint64_t*> NTTEngine::d_invnthroot;
std::map<int, uint64_t*> NTTEngine::h_nthroot;
std::map<int, uint64_t*> NTTEngine::h_invnthroot;
uint64_t *NTTEngine::d_gN = NULL;
uint64_t *NTTEngine::d_ginvN = NULL;

extern __constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];

// These macros improves readability at execute_ntt
#define runNTTKernel(N)                                   \
  if(direction == FORWARD)                                \
    ntt<N><<< CUDAManager::get_n_residues(data->base), (N >> 1), 0, ctx->get_stream()>>>(    \
      data->d_coefs,                                               \
      CUDAManager::get_n_residues(data->base),                                          \
      hob(N),                                             \
      COMMONEngine::d_gjk[N],                                \
      NTTEngine::d_nthroot[N]);                           \
  else                                                    \
    intt<N><<< CUDAManager::get_n_residues(data->base), (N >> 1), 0, ctx->get_stream()>>>(\
      data->d_coefs,                                               \
      CUDAManager::get_n_residues(data->base),                                          \
      hob(N),                                             \
      COMMONEngine::d_invgjk[N],                             \
      NTTEngine::d_invnthroot[N]);        

#define runNTTKernelHierarchical(Na, Nb)                                      \
    if(direction == FORWARD){                                                 \
    hntt_tr<Nb><<< Na * CUDAManager::get_n_residues(data->base), Nb/2, 0, ctx->get_stream()>>>(           \
      ctx->d_tmp,                                                          \
      data->d_coefs,                                                                     \
      CUDAManager::get_n_residues(data->base),                                                                \
      hob(Nb),                                                                  \
      NTTEngine::d_nthroot[Na * Nb],             \
      COMMONEngine::d_gjk[Nb]);                                  \
    hntt_tr<Na><<< Nb * CUDAManager::get_n_residues(data->base), Na/2, 0, ctx->get_stream()>>>(           \
      data->d_coefs,                                                                     \
      ctx->d_tmp,                                                          \
      CUDAManager::get_n_residues(data->base),                                                                \
      hob(Na),                                                                  \
      NTTEngine::d_gN,                             \
      COMMONEngine::d_gjk[Na]);                                  \
  }else{                                                                        \
    hintt<Na><<< Nb * CUDAManager::get_n_residues(data->base), Na/2, 0, ctx->get_stream()>>>(             \
      ctx->d_tmp,                                                          \
      data->d_coefs,                                                                     \
      CUDAManager::get_n_residues(data->base),                                                                \
      hob(Na),                                                                  \
      NTTEngine::d_ginvN,                          \
      COMMONEngine::d_invgjk[Na]);                                  \
    hintt_tr<Nb><<< Na * CUDAManager::get_n_residues(data->base), Nb/2, 0, ctx->get_stream()>>>(          \
      data->d_coefs,                                                                     \
      ctx->d_tmp,                                                          \
      CUDAManager::get_n_residues(data->base),                                                                \
      hob(Nb),                                                                  \
      NTTEngine::d_invnthroot[Na * Nb],          \
      COMMONEngine::d_invgjk[Nb]);                                  \
    }     

#define runNTT(N)   \
  switch(N){        \
    case 8:        \
    runNTTKernel(8);\
    break;          \
    case 16:        \
    runNTTKernel(16);\
    break;          \
    case 32:        \
    runNTTKernel(32);  \
    break;          \
    case 64:        \
    runNTTKernel(64);  \
    break;          \
    case 128:       \
    if(ENABLE_HNTT_128) runNTTKernelHierarchical(NTTFACT128a, NTTFACT128b) else runNTTKernel(128); \
    break;          \
    case 256:       \
    if(ENABLE_HNTT_256) runNTTKernelHierarchical(NTTFACT256, NTTFACT256) else runNTTKernel(256); \
    break;          \
    case 512:       \
    if(ENABLE_HNTT_512) runNTTKernelHierarchical(NTTFACT512a, NTTFACT512b) else runNTTKernel(512); \
    break;          \
    case 1024:      \
    if(ENABLE_HNTT_1024) runNTTKernelHierarchical(NTTFACT1024, NTTFACT1024) else runNTTKernel(1024); \
    break;          \
    case 2048:      \
    if(ENABLE_HNTT_2048) runNTTKernelHierarchical(NTTFACT2048a, NTTFACT2048b) else runNTTKernel(2048); \
    break;          \
    case 4096:      \
    runNTTKernelHierarchical(NTTFACT4096, NTTFACT4096); \
    break;          \
    case 8192:      \
    runNTTKernelHierarchical(NTTFACT8192a, NTTFACT8192b);\
    break;          \
    case 16384:      \
    runNTTKernelHierarchical(NTTFACT16384, NTTFACT16384);\
    break;          \
    case 32768:      \
    runNTTKernelHierarchical(NTTFACT32768a, NTTFACT32768b);\
    break;          \
    case 65536:      \
    runNTTKernelHierarchical(NTTFACT65536, NTTFACT65536);\
    break;          \
    case 131072:      \
    runNTTKernelHierarchical(NTTFACT131072a, NTTFACT131072b);\
    break;          \
    case 262144:      \
    runNTTKernelHierarchical(NTTFACT262144, NTTFACT262144);\
    break;          \
    case 524288:      \
    runNTTKernelHierarchical(NTTFACT524288a, NTTFACT524288b);\
    break;          \
    case 1048576:      \
    runNTTKernelHierarchical(NTTFACT1048576, NTTFACT1048576);\
    break;          \
    default:        \
    throw std::runtime_error("NTT is not working for this N");\
  }

__device__ void ntttransform(
  uint64_t* data,
  const int cid,
  const int rid,
  const int stride,
  const int N,
  const int nresidues,
  const int m,
  const int log2N,
  const uint64_t *gjk){

  // Indexes
  const int j = cid * 2 * m / N;
  const int i = j + (cid % (N/(2*m)))*2*m;
  const int xi_index = i;
  const int xim_index = i + m;

  // Coefs
  uint64_t xi = data[xi_index];
  uint64_t xim = data[xim_index];

  // DGT
  const uint64_t a = gjk[(rid * log2N + stride) * N + j];
  
  // Write the result
  data[xi_index]    = addmod(xi, xim, rid); 
  uint64_t xisubxim = submod(xi, xim, rid);
  data[xim_index] = mulmod(a, xisubxim, rid);

  return;
}

__device__ void intttransform(
  uint64_t* data,
  const int cid,
  const int rid,
  const int stride,
  const int N,
  const int nresidues,
  const int m,
  const int log2N,
  const uint64_t *invgjk){

  // Indexes
  const int j = cid * 2 * m / N;
  const int i = j + (cid % (N/(2*m)))*2*m;
  const int xi_index = i;
  const int xim_index = i + m;

  // Coefs
  uint64_t xi = data[xi_index];
  uint64_t xim = data[xim_index];
  uint64_t new_xi, new_xim;

  // IDGT
  const uint64_t a = invgjk[(rid * log2N + stride) * N + j];
  const uint64_t axim = mulmod(a, xim, rid);
 
  new_xi = addmod(xi, axim, rid); 
  new_xim = submod(xi, axim, rid); 

  // Write the result
  data[xi_index] = new_xi;
  data[xim_index] = new_xim;

  return;
}

template <int N>
__device__ void doNTT(
  uint64_t* data,
  int cid,
  int rid,
  const int nresidues,
  const uint64_t *gjk){

  int log2N = bitsize_int(N);
  for(int stride = 0; stride < log2N; stride++){
    int m = N / (2<<stride);
    ntttransform(
      data,
      cid,
      rid,
      stride,
      N,
      nresidues,
      m,
      log2N,
      gjk);
    __syncthreads();
  }
}


template <int N>
__device__ void doINTT(
  uint64_t* data,
  int cid,
  int rid,
  const int nresidues,
  const uint64_t *invgjk){

  int log2N = bitsize_int(N);
  for(int stride = log2N - 1; stride >= 0; stride--){
    int m = N / (2<<stride);
    intttransform(
      data,
      cid,
      rid,
      stride,
      N,
      nresidues,
      m,
      log2N,
      invgjk);
    __syncthreads();
  }

}


// Standard gentleman-sande NTT
template <int N>
__global__ void ntt(
  uint64_t* data,
  const int nresidues,
  const int log2k,
  const uint64_t *gjk,
  const uint64_t *nthroots){

  const int rid = blockIdx.x;
  const int index_offset = blockIdx.x * N; 
  __shared__ uint64_t s_data[N];

  s_data[threadIdx.x] = data[threadIdx.x + index_offset];
  s_data[threadIdx.x + (N >> 1)] = data[threadIdx.x + (N >> 1) + index_offset];

  __syncthreads();

  // Twist the folded signal
  s_data[threadIdx.x]            = mulmod(s_data[threadIdx.x],        nthroots[threadIdx.x         + rid * N], rid); // Twist
  s_data[threadIdx.x + (N >> 1)] = mulmod(s_data[threadIdx.x + (N >> 1)],  nthroots[(threadIdx.x + (N >> 1)) + rid * N], rid); // Twist


  // NTT
  doNTT<N>(s_data, threadIdx.x, rid, nresidues, gjk);

  // Outcome
  data[threadIdx.x + index_offset] = s_data[threadIdx.x];
  data[threadIdx.x + (N >> 1) + index_offset] = s_data[threadIdx.x + (N >> 1)];
}

// Standard cooley-tukey INTT
template <int N>
__global__ void intt(
  uint64_t* data,
  const int nresidues,
  const int log2k,
  const uint64_t *invgjk,
  const uint64_t *invnthroots){

  const int rid = blockIdx.x;
  const int index_offset = rid * N; // rid_offset does not affect this
  __shared__ uint64_t s_data[N];

  s_data[threadIdx.x] = data[threadIdx.x + index_offset];
  s_data[threadIdx.x + (N >> 1)] = data[threadIdx.x + (N >> 1) + index_offset];
  __syncthreads();
  
  // INTT
  doINTT<N>(s_data, threadIdx.x, rid, nresidues, invgjk);

  // "Untwist" the folded signal
  s_data[threadIdx.x]            = mulmod(s_data[threadIdx.x],        invnthroots[threadIdx.x         + rid * N], rid); // "Untwist"
  s_data[threadIdx.x + (N >> 1)] = mulmod(s_data[threadIdx.x + (N >> 1)],  invnthroots[(threadIdx.x + (N >> 1)) + rid * N], rid); // "Untwist"


  // Outcome
  data[threadIdx.x + index_offset]       = s_data[threadIdx.x];
  data[threadIdx.x + (N >> 1) + index_offset] = s_data[threadIdx.x + (N >> 1)];
}

/*NTT "along the rows" 
data is treated as a Na x Nb matrix (row-major)
perform Nc NTTs of size 2 * blockDim.x "along the rows", I mean data is 
read through rows.

This kernel applies the NTT on columns. The outcome is written transposed.
Because of this we need to read and write on different arrays, otherwise 
we would create a race condition.

The template parameter is used to instantiate s_data.
*/
template <int Nblock>
__global__ void hntt_tr(
  uint64_t* odata,
  uint64_t* idata,
  const int nresidues,
  const int log2k,
  const uint64_t *C,
  const uint64_t *gjk){

  const int Nr = blockDim.x * 2; // Number of columns
  const int Nc = gridDim.x / nresidues; // Number of rows
  const int rid = (blockIdx.x  / Nc);
  const int index_offset  = rid * Nc * Nr; // offset to target a particular residue
 
  const int in_xi_index   = threadIdx.x                * Nc + blockIdx.x % Nc;
  const int in_xim_index  = (threadIdx.x + blockDim.x) * Nc + blockIdx.x % Nc;
  const int out_xi_index  = threadIdx.x                     + blockIdx.x * Nr;
  const int out_xim_index = (threadIdx.x + blockDim.x)      + blockIdx.x * Nr; // write transposed
 
  __shared__ uint64_t s_data[Nblock];

  s_data[threadIdx.x]              = idata[in_xi_index + index_offset];
  s_data[threadIdx.x + blockDim.x] = idata[in_xim_index + index_offset];

  // Twist
  s_data[threadIdx.x]              = mulmod(
    s_data[threadIdx.x],
    C[in_xi_index + index_offset],
    rid);
  s_data[threadIdx.x + blockDim.x] = mulmod(
    s_data[threadIdx.x + blockDim.x],
    C[in_xim_index + index_offset],
    rid);

  __syncthreads();

  // NTT
  doNTT<Nblock>(s_data, threadIdx.x, rid, nresidues, gjk);

  // Outcome
  odata[out_xi_index]  = s_data[threadIdx.x];
  odata[out_xim_index] = s_data[threadIdx.x + blockDim.x];
}

/*INTT "along the rows"

data is treated as a Na x Nb matrix (row-major)
perform Nc INTTs of size 2 * blockDim.x "along the columns".

This kernel applies the INTT on rows. 
*/
template <int Nblock>
__global__ void hintt(
  uint64_t* out_data,
  uint64_t* data,
  const int nresidues,
  const int log2k,
  const uint64_t *C,
  const uint64_t *invgjk){

  const int rid = (blockIdx.x / (gridDim.x / nresidues));
  const int index_offset = blockIdx.x * Nblock; // offset to choose the residue to read
  
  const int xi_index = threadIdx.x;
  const int xim_index = threadIdx.x + blockDim.x;
  
  __shared__ uint64_t s_data[Nblock];

  s_data[xi_index]  = data[xi_index  + index_offset];
  s_data[xim_index] = data[xim_index + index_offset];
  
  __syncthreads();
  
  // INTT
  doINTT<Nblock>(s_data, threadIdx.x, rid, nresidues, invgjk);

  // Untwist
  s_data[xi_index]  = mulmod(
    s_data[xi_index],
    C[xi_index  + index_offset],
    rid);
  s_data[xim_index] = mulmod(
    s_data[xim_index],
    C[xim_index + index_offset],
    rid);

  // Outcome
  out_data[xi_index + index_offset]  = s_data[xi_index];
  out_data[xim_index + index_offset] = s_data[xim_index];
}

/*INTT "along the rows" 
data is treated as a Na x Nb matrix (row-major)
perform Nc INTTs of size 2 * blockDim.x "along the rows".

This kernel applies the INTT on columns. Different than hdgt_tr, the outcome is
NOT written transposed.
*/
template <int Nblock>
__global__ void hintt_tr(
  uint64_t* odata,
  uint64_t* idata,
  const int nresidues,
  const int log2k,
  const uint64_t *C,
  const uint64_t *invgjk){

  const int Nr = blockDim.x * 2; // Number of columns
  const int Nc = gridDim.x / nresidues; // Number of rows
  const int rid = (blockIdx.x / Nc);
  const int index_offset = rid * Nc * Nr; // offset to specify the residue to read

  const int in_xi_index  =  threadIdx.x                * Nc + blockIdx.x % Nc;
  const int in_xim_index = (threadIdx.x + blockDim.x)  * Nc + blockIdx.x % Nc;
  const int out_xi_index  =  in_xi_index;
  const int out_xim_index =  in_xim_index;
 
  __shared__ uint64_t s_data[Nblock];

  s_data[threadIdx.x]                = idata[in_xi_index + index_offset];
  s_data[(threadIdx.x + blockDim.x)] = idata[in_xim_index + index_offset];

  __syncthreads();
    
  doINTT<Nblock>(s_data, threadIdx.x, rid, nresidues, invgjk);

  // Untwist
  s_data[threadIdx.x]  = mulmod(
    s_data[threadIdx.x],
    C[out_xi_index  + index_offset],
    rid);
  s_data[threadIdx.x + blockDim.x] = mulmod(
    s_data[threadIdx.x + blockDim.x],
    C[out_xim_index + index_offset],
    rid);

  // Outcome
  odata[out_xi_index  + index_offset]  = s_data[threadIdx.x];
  odata[out_xim_index + index_offset] = s_data[threadIdx.x + blockDim.x];
}


void NTTEngine::execute_ntt(
  Context *ctx,
  poly_t *data,
  const transform_directions direction){ // Forward or Inverse

  if(data->state == TRANSSTATE && direction == FORWARD ||
    data->state == RNSSTATE && direction == INVERSE)
    return;

  runNTT(CUDAManager::N);
  
  data->state = (direction == FORWARD? TRANSSTATE : RNSSTATE);
}

///////////////////////////////////////////////////////////////////////////////

__global__ void ntt_vector_op(
  uint64_t *c,
  const uint64_t *a,
  const uint64_t *b,
  const supported_operations OP,
  const int N, const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;

  if(tid < N * NResidues){
    switch(OP){
    case ADDOP:
    c[tid] = addmod(a[tid], b[tid], rid);
    break;
    case SUBOP:
    c[tid] = submod(a[tid], b[tid], rid);
    break;
    case MULOP:
    c[tid] = mulmod(a[tid], b[tid], rid);
    break;
    case NEGATEOP:
    c[tid] = (d_RNSCoprimes[rid] - a[tid]) * (a[tid] != 0);
    break;
    default:
    printf("ntt_vector_op: Unknown operation %d\n", OP);
    break;
    }
  }
}

void NTTEngine::execute_op(
  Context *ctx,
  uint64_t *c,
  const uint64_t *a,
  const uint64_t *b,
  const supported_operations OP,
  const poly_bases base){

  const int size = CUDAManager::N * CUDAManager::get_n_residues(base);
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  ntt_vector_op<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    c, a, b,
    OP,
    CUDAManager::N, CUDAManager::get_n_residues(base));
  cudaCheckError();

}

__global__ void ntt_vector_op_int(
  uint64_t *c,
  const uint64_t *a,
  const uint64_t b,
  const supported_operations OP,
  const int N, const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;

  if(tid < N * NResidues){
    switch(OP){
    case ADDOP:
    c[tid] = addmod(a[tid], b, rid);
    break;
    case SUBOP:
    c[tid] = submod(a[tid], b, rid);
    break;
    case MULOP:
    c[tid] = mulmod(a[tid], b, rid);
    break;
    default:
    printf("ntt_vector_op_int: Unknown operation %d\n", OP);
    break;
    }
  }
}

void NTTEngine::execute_op_by_uint(
  Context *ctx,
  uint64_t *c,
  const uint64_t *a,
  const uint64_t b,
  const supported_operations OP,
  const poly_bases base){

  const int size = CUDAManager::N * CUDAManager::get_n_residues(base);
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  ntt_vector_op_int<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    c,
    a,
    b,
    OP,
    CUDAManager::N, CUDAManager::get_n_residues(base));
  cudaCheckError();

}

__global__ void ntt_vector_dualop(
  uint64_t *c, uint64_t *a, uint64_t *b,
  uint64_t *f, uint64_t *d, uint64_t *e,
  const supported_operations OP,
  const int N, const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;

  if(tid < N * NResidues){
    switch(OP){
    case ADDADDOP:
    c[tid] = addmod(a[tid], b[tid], rid);
    f[tid] = addmod(d[tid], e[tid], rid);
    break;
    default:
    printf("ntt_vector_dualop: Unknown operation %d\n", OP);
    break;
    }
  }
}

void NTTEngine::execute_dualop(
  Context *ctx,
  uint64_t *c, uint64_t *a, uint64_t *b,
  uint64_t *f, uint64_t *d, uint64_t *e,
  const supported_operations OP,
  const poly_bases base){

  const int size = CUDAManager::N * CUDAManager::get_n_residues(base);
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  ntt_vector_dualop<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    c, a, b,
    f, d, e,
    OP,
    CUDAManager::N, CUDAManager::get_n_residues(base));
  cudaCheckError();

}

__global__ void ntt_vector_seqop(
  uint64_t *d,
  const uint64_t *a,
  const uint64_t *b,
  const uint64_t *c,
  const supported_operations OP,
  const int N, const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;

  if(tid < N * NResidues){
    // switch(OP){
    // case MULandADDOP:
    uint64_t aux = mulmod(a[tid], b[tid], rid);
    d[tid] = addmod(aux, c[tid], rid);
    // break;
    // default:
    // printf("Unknown operation %d\n", OP);
    // break;
    // }
  }
}

void NTTEngine::execute_seqop(
  Context *ctx,
  uint64_t *d,
  uint64_t *a, uint64_t *b, uint64_t *c,
  const supported_operations OP,
  const poly_bases base){

  const int size = CUDAManager::N * CUDAManager::get_n_residues(base);
  const dim3 gridDim(get_grid_dim(size, DEFAULTBLOCKSIZE));
  const dim3 blockDim(DEFAULTBLOCKSIZE);

  ntt_vector_seqop<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    d, a, b, c,
    OP,
    CUDAManager::N, CUDAManager::get_n_residues(base));
  cudaCheckError();

}

///////////////////////////////////////////////////////////////////////////////


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
void compute_nthroot_ntt(
  uint64_t *o_d_nthroots,
  uint64_t *o_d_invnthroots,
  uint64_t *o_h_nthroots,
  uint64_t *o_h_invnthroots,
  int signalsize,
  bool isHNTT = false){

  // Alloc
  uint64_t *h_nthroots = (uint64_t*) calloc ( 
    signalsize * CUDAManager::get_n_residues(QBBase), 
    sizeof(uint64_t));
  uint64_t *h_invnthroots = (uint64_t*) calloc ( 
    signalsize * CUDAManager::get_n_residues(QBBase), 
    sizeof(uint64_t));

  // Compute the powers of the root
  for(
    int rid = 0;
    rid < CUDAManager::get_n_residues(QBBase);
    rid++)
    for(int cid = 0; cid < signalsize; cid++){
      uint64_t p = COPRIMES_BUCKET[rid];
      uint64_t psi = fast_pow((uint64_t)PROOTS[p], (p-1) / (2 * signalsize), rid);
      uint64_t invpsi = conv<uint64_t>(NTL::InvMod(to_ZZ(psi), to_ZZ(p)));

      assert(fast_pow(psi, 2 * signalsize, rid) == 1);

      h_nthroots   [cid + rid * signalsize] = fast_pow(psi, cid, rid);
      if(!isHNTT) // Normal NTT
        h_invnthroots[cid + rid * signalsize] = mulmod(
        fast_pow(invpsi, cid, rid),
        conv<uint64_t>(NTL::InvMod(to_ZZ(signalsize), to_ZZ(p))), // Embed the scaling factor
        rid);
      else // Hierarchical NTT
        h_invnthroots[cid + rid * signalsize] = fast_pow(invpsi, cid, rid);
    }
  
  // Copy the powers of the root
  cudaMemcpy(
    o_d_nthroots,
    h_nthroots,
    signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError()
  cudaMemcpy(
    o_d_invnthroots,
    h_invnthroots,
    signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError()
  memcpy(
    o_h_nthroots,
    h_nthroots,
    signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t));
  memcpy(
    o_h_invnthroots,
    h_invnthroots,
    signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t));

  // Release temporary memory
  free(h_nthroots);
  free(h_invnthroots);
}

__host__ void NTTEngine::init(){
  
  // Common assertions
  assert(CUDAManager::is_init);
  assert(CUDAManager::N > 0);
  
  // Alloc
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaCheckError();

  // bitreverse
  // We use CUDAManager::N as upper bound
  int* h_bitreversalmap_Na = (int*) calloc (CUDAManager::N, sizeof(int)); 
  int* h_bitreversalmap_Nb = (int*) calloc (CUDAManager::N, sizeof(int));

  // powers of g
  uint64_t *h_gN = (uint64_t*) calloc ( 
    CUDAManager::N * CUDAManager::get_n_residues(QBBase), 
    sizeof(uint64_t));
  uint64_t *h_ginvN = (uint64_t*) calloc ( 
    CUDAManager::N * CUDAManager::get_n_residues(QBBase), 
    sizeof(uint64_t));
  
  cudaMalloc(
    (void**)&d_gN,
    poly_get_size(QBBase));
  cudaCheckError();

  cudaMalloc(
    (void**)&d_ginvN,
    poly_get_size(QBBase));
  cudaCheckError();

  /* If hierarchical NTT we need to pre-compute these powers for different 
  signl sizes (64, 128, ...). Otherwise, we only need one 
  signal size = CUDAManager::N */
  std::vector<int> signalsizes {};
  bool isHNTT = false;
  int Na = -1, Nb = -1;
  switch(CUDAManager::N){
    case 128:
    Na = NTTFACT128a;
    Nb = NTTFACT128b;

    signalsizes.push_back(Na);
    signalsizes.push_back(Nb);
    isHNTT = ENABLE_HNTT_128;
    break;
    case 256:
    Na = NTTFACT256;
    Nb = NTTFACT256;

    signalsizes.push_back(Na);
    isHNTT = ENABLE_HNTT_256;
    break;
    case 512:
    Na = NTTFACT512a;
    Nb = NTTFACT512b;

    signalsizes.push_back(Na);
    signalsizes.push_back(Nb);
    isHNTT = ENABLE_HNTT_512;
    break;
    case 1024:
    Na = NTTFACT1024;
    Nb = NTTFACT1024;

    signalsizes.push_back(Na);
    isHNTT = ENABLE_HNTT_1024;
    break;
    case 2048:
    Na = NTTFACT2048a;
    Nb = NTTFACT2048b;

    signalsizes.push_back(Na);
    signalsizes.push_back(Nb);
    isHNTT = ENABLE_HNTT_2048;
    break;
    case 4096:
    Na = NTTFACT4096;
    Nb = NTTFACT4096;

    signalsizes.push_back(Na);
    isHNTT = true;
    break;
    case 8192:
    Na = NTTFACT8192a;
    Nb = NTTFACT8192b;

    signalsizes.push_back(Na);
    signalsizes.push_back(Nb);
    isHNTT = true;
    break;
    case 16384:
    Na = NTTFACT16384;
    Nb = NTTFACT16384;

    signalsizes.push_back(Na);
    isHNTT = true;
    break;
    case 32768:
    Na = NTTFACT32768a;
    Nb = NTTFACT32768b;

    signalsizes.push_back(Na);
    signalsizes.push_back(Nb);
    isHNTT = true;
    break;
    case 65536:
    Na = NTTFACT65536;
    Nb = NTTFACT65536;

    signalsizes.push_back(Na);
    isHNTT = true;
    break;
    case 131072:
    Na = NTTFACT131072a;
    Nb = NTTFACT131072b;

    signalsizes.push_back(Na);
    signalsizes.push_back(Nb);
    isHNTT = true;
    break;
    case 262144:
    Na = NTTFACT262144;
    Nb = NTTFACT262144;

    signalsizes.push_back(Na);
    isHNTT = true;
    break;
    case 524288:
    Na = NTTFACT524288a;
    Nb = NTTFACT524288b;

    signalsizes.push_back(Na);
    signalsizes.push_back(Nb);
    isHNTT = true;
    break;
    case 1048576:
    Na = NTTFACT1048576;
    Nb = NTTFACT1048576;

    signalsizes.push_back(Na);
    isHNTT = true;
    break;
  }
  signalsizes.push_back(CUDAManager::N);

  for(unsigned int i = 0; i < signalsizes.size(); i++){
    int signalsize = signalsizes[i];

    uint64_t *gjk_aux;

    if(COMMONEngine::d_gjk.find(signalsize) != COMMONEngine::d_gjk.end()) // Already computed
      continue;

    cudaMalloc(
      (void**)&gjk_aux,
      CUDAManager::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t)
      );
    cudaCheckError();
    COMMONEngine::d_gjk[signalsize] = gjk_aux;

    cudaMalloc(
      (void**)&gjk_aux,
      CUDAManager::get_n_residues(QBBase) * ilog2(signalsize) * signalsize * sizeof(uint64_t)
      );
    cudaCheckError();
    COMMONEngine::d_invgjk[signalsize] = gjk_aux;

    // pre-compute
    compute_gjk(
      COMMONEngine::d_gjk[signalsize],
      COMMONEngine::d_invgjk[signalsize],
      signalsize);
  }

  for(unsigned int i = 0; i < signalsizes.size(); i++){
    int signalsize = signalsizes[i];

    uint64_t *nthroot_aux;
    cudaMalloc(
      (void**)&nthroot_aux,
      signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t)
      );
    cudaCheckError();
    d_nthroot[signalsize] = nthroot_aux;

    cudaMalloc(
      (void**)&nthroot_aux,
      signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t)
      );
    cudaCheckError();
    d_invnthroot[signalsize] = nthroot_aux;

    h_nthroot[signalsize] = (uint64_t*) malloc(
      signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t));
    h_invnthroot[signalsize] = (uint64_t*) malloc(
      signalsize * CUDAManager::get_n_residues(QBBase) * sizeof(uint64_t));

    // pre-compute
    compute_nthroot_ntt(
      d_nthroot[signalsize],
      d_invnthroot[signalsize],
      h_nthroot[signalsize],
      h_invnthroot[signalsize],
      signalsize,
      isHNTT);
  }

  if(isHNTT){
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
      rid < CUDAManager::get_n_residues(QBBase);
      rid++){
      uint64_t p = COPRIMES_BUCKET[rid];
      uint64_t n = (p-1)/(CUDAManager::N);
      assert((p-1) % (CUDAManager::N) == 0);

      uint64_t g = fast_pow((uint64_t)PROOTS[p], n, rid);
      uint64_t g_inv = conv<uint64_t>(NTL::InvMod(to_ZZ(g), to_ZZ(p)));

      assert(mulmod(g, g_inv, rid) % p == 1);
      assert(Na && Nb);
      for(int cid = 0; cid < (CUDAManager::N); cid++){
        h_gN   [cid + rid * (CUDAManager::N)] = fast_pow(
          g,
          (cid / Nb) * h_bitreversalmap_Nb[cid % Nb],
          rid);
        h_ginvN[cid + rid * (CUDAManager::N)] = mulmod(
            fast_pow(
            g_inv,
            h_bitreversalmap_Nb[cid / Na] * (cid % Na),
            rid),
            conv<uint64_t>(NTL::InvMod(to_ZZ(Na * Nb), to_ZZ(COPRIMES_BUCKET[rid]))), // Embed the scaling
            rid);
      }
    }

    // Copy the powers of the root
    cudaMemcpyAsync( d_gN, h_gN, poly_get_size(QBBase), cudaMemcpyHostToDevice, stream );
    cudaCheckError()
    cudaMemcpyAsync( d_ginvN, h_ginvN, poly_get_size(QBBase), cudaMemcpyHostToDevice, stream );
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
  #ifdef ENABLE_DEBUG_MACRO
  std::cout << "NTT initialized." << std::endl;
  #endif
}

__host__ void NTTEngine::destroy(){
  if(!is_init)
    return;

  cudaDeviceSynchronize();
  cudaCheckError();

  cudaFree(d_gN);
  cudaCheckError();
  cudaFree(d_ginvN);
  cudaCheckError();

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

  d_nthroot.clear();
  d_invnthroot.clear();
  h_nthroot.clear();
  h_invnthroot.clear();

  is_init = false;
}