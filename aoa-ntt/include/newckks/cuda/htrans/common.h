#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <NTL/ZZ.h>
#include <NTL/RR.h>
#include <map>
// #include <newckks/coprimes.h>
#include <newckks/tool/context.h>
#include <newckks/defines.h>

#define WORDSIZE (int)64

/**
 * @brief      Macro for checking cuda errors following a cuda launch or api call
 *
 */
#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if( e == cudaErrorInvalidDevicePointer)   \
   fprintf(stderr, "Cuda failure %s:%d: '%s' (%d)\n",__FILE__,__LINE__,cudaGetErrorString(e), e);           \
 else if(e != cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s' (%d)\n",__FILE__,__LINE__,cudaGetErrorString(e), e);           \
    exit(1);                                                                 \
 }                                                                      \
}

#define hasSupportStreamAlloc()({int r; cudaDeviceGetAttribute(&r, cudaDevAttrMemoryPoolsSupported, 0); r;})

inline bool operator==(const GaussianInteger& a, const GaussianInteger& b){
    return (a.re == b.re) && (a.imag == b.imag);
};

__host__ int hob (int num);

__host__ int ilog2(int n);

__host__ __device__ bool operator==(const uint128_t &a, const uint128_t &b);

__host__ __device__ bool operator!=(const uint128_t& a, const uint128_t& b);

__host__ __device__ bool operator<(const uint128_t &a, const uint128_t &b);

__host__ __device__ bool operator>(const uint128_t &a, const uint128_t &b);

__host__ __device__ bool operator<=(const uint128_t &a, const uint128_t &b);

__host__ __device__ bool operator>=(const uint128_t &a, const uint128_t &b);

__host__ __device__ bool operator>=(const uint128_t &a, const uint64_t &b);

__host__ __device__ uint128_t operator-(const uint128_t &a, const uint128_t &b);

__host__ __device__ inline uint128_t lshift(uint128_t x, int b);

__host__ __device__ inline uint128_t rshift(uint128_t x, int b);

__host__ __device__ inline uint64_t mul64hi(uint64_t a, uint64_t b) ;

__host__ __device__ inline void load_coprime(uint64_t *p, int *numbits, int rid);

__host__ __device__  uint128_t mullazy(const uint64_t a, const uint64_t b);

__host__ __device__ uint128_t add128(uint128_t x, uint128_t y, int rid); // x + y

__host__ __device__ uint128_t sub128(uint128_t x, uint128_t y, int rid); // x - y

__host__ __device__ uint64_t mod(uint128_t a, int rid);

__host__ __device__  uint64_t mulmod(const uint64_t a, const uint64_t b, const int rid);

__host__ __device__ uint64_t fast_pow(uint64_t a, uint64_t b, int rid);


/**
 * @brief       Returns A if C == 0 and B if C == 1
 *
 */
#define SEL(A, B, C) ((-(C) & ((A) ^ (B))) ^ (A))
#ifdef __CUDA_ARCH__
#define addmod(a, b, rid) SEL(a+b,a+b-d_RNSCoprimes[rid],(a+b) >= d_RNSCoprimes[rid])
#define submod(a, b, rid) SEL(a-b,a+(d_RNSCoprimes[rid]-b),a<b)
#else
#define addmod(a, b, rid) SEL(a+b,a+b-COPRIMES_BUCKET[rid],(a+b) >= COPRIMES_BUCKET[rid])
#define submod(a, b, rid) SEL(a-b,a+(COPRIMES_BUCKET[rid]-b),a<b)
#endif

// Returns the bit-size of a
#ifdef __CUDA_ARCH__
#define bitsize_int(a) (31 - __clz(a))
#else
#define bitsize_int(a) (31 - __builtin_clzl(a))
#endif

void compute_gjk( uint64_t *o_gjk, uint64_t *o_invgjk, int signalsize);

class COMMONEngine{
	public:
    static bool is_init; //!< True if is initialized and ready for use. Otherwise, False.
  	static std::map<int, uint64_t*> d_gjk; //!< Device array which stores the powers of \f$g\f$ for the DGT.
  	static std::map<int, uint64_t*> d_invgjk; //!< Device array which stores the powers of \f$g^{-1}\f$ for the DGT.

    static uint64_t *d_tmp_data;
    static engine_types engine_mode;
    /**
     * @brief Initializes the related engine.
     */
	static void init(engine_types e);
  
    /*! \brief Destroy the object.
    *
    * Deallocate all related data on the device and host memory.
    */    
	static void destroy();

    /**
     * Applies a transform
     * 
     * @param ctx       [description]
     * @param data      [description]
     * @param direction [description]
     * @param base      [description]
     */
    static void execute(
        Context *ctx,
        poly_t *data,
        const transform_directions direction); // Forward or Inverse

    static void execute_op(
        Context *ctx,
        uint64_t *c,
        uint64_t *a,
        uint64_t *b,
        const supported_operations OP,
        const poly_bases base);

    static void execute_op_by_uint(
        Context *ctx,
        uint64_t *c,
        uint64_t *a,
        const uint64_t b,
        const supported_operations OP,
        const poly_bases base);

    static void execute_dualop(
        Context *ctx,
        uint64_t *c,
        uint64_t *a,
        uint64_t *b,
        uint64_t *f,
        uint64_t *d,
        uint64_t *e,
        const supported_operations OP,
        const poly_bases base);

    static void execute_seqop(
        Context *ctx,
        uint64_t *d,
        uint64_t *a,
        uint64_t *b,
        uint64_t *c,
        const supported_operations OP,
        const poly_bases base);

};

#endif