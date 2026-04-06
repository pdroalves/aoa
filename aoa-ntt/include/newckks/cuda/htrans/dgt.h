#ifndef DGT_H
#define DGT_H
#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <map>
#include <sstream>      // std::ostringstream
#include <newckks/cuda/htrans/common.h>
#include <newckks/cuda/manager.h>

// Flags to switch between DGT and HDGT
#define ENABLE_HDGT_128 true
#define ENABLE_HDGT_256 true
#define ENABLE_HDGT_512 true
#define ENABLE_HDGT_1024 true
#define ENABLE_HDGT_2048 true

////////////////////////////////////////////////////////////////////////////////
// Typedefs and definitions
////////////////////////////////////////////////////////////////////////////////

// These are the factorizations used on DGT hierarchical for each N
#define DGTFACT64 (int)8
#define DGTFACT128a (int)16
#define DGTFACT128b (int)8
#define DGTFACT256 (int)16
#define DGTFACT512a (int)32
#define DGTFACT512b (int)16
#define DGTFACT1024 (int)32
#define DGTFACT2048a (int)64
#define DGTFACT2048b (int)32
#define DGTFACT4096  (int)64
#define DGTFACT8192a (int)128
#define DGTFACT8192b (int)64
#define DGTFACT16384 (int)128
#define DGTFACT32768a (int)256
#define DGTFACT32768b (int)128
#define DGTFACT65536 (int)256
#define DGTFACT131072a (int)512
#define DGTFACT131072b (int)256
#define DGTFACT262144 (int)512
#define DGTFACT524288a (int)1024
#define DGTFACT524288b (int)512
#define DGTFACT1048576 (int)1024

__host__ __device__ GaussianInteger GIAdd(GaussianInteger a, GaussianInteger b, int rid);
__host__ __device__ GaussianInteger GISub(GaussianInteger a, GaussianInteger b, int rid);
__device__ void mulmod_gi(GaussianInteger *c, const GaussianInteger a, const uint64_t b, const int rid);
__device__ GaussianInteger mulmod_gi(const GaussianInteger a, const uint64_t b, const int rid);

__global__ void convert_U64_to_GI(GaussianInteger *b, const uint64_t *a, int N, int nresidues);
__global__ void convert_GI_to_U64(uint64_t *b, GaussianInteger *a, int N, int nresidues);

/**
 * @brief      All operations related to DGT are contained in this class.
 *             It follows the Singleton design pattern.
 *             
 *             All variables are expected to be read-only after initilization.
 */
class DGTEngine{
  public:

    static bool is_init; //!< True if is initialized and ready for use. Otherwise, False.

    //////////////////
    //   Parameters //
    //////////////////
    //
    static uint64_t *d_gN; //!< Device array which stores the powers of \f$g\f$ for the second step of the hierarchical NTT.
    static uint64_t *d_ginvN; //!< Device array which stores the powers of \f$g^{-1}\f$ for the second step of the hierarchical NTT.
    static std::map<int, GaussianInteger*> d_nthroot; //!< Powers of n-th root of i for different signal sizes (device).
    static std::map<int, GaussianInteger*> d_invnthroot; //!< Powers of the inverse of n-th root of i for different signal sizes (device).
    static std::map<int, GaussianInteger*> h_nthroot; //!< Powers of n-th root of i for different signal sizes (host).
    static std::map<int, GaussianInteger*> h_invnthroot;//!< Powers of the inverse of n-th root of i for different signal sizes (host).

    /**
     * @brief Initializes DGTEngine.
     *
     * Allocates and initializes all the data structure used by DGTEngine's methods.
     * Pre-computes g^j and its modular inverse.
     */
    static void init();
  
    /*! \brief Destroy the object.
    *
    * Deallocate all related data on the device and host memory.
    */    
    static void destroy();

    static void execute_dgt(
      Context *ctx,
      poly_t *data,
      const transform_directions direction,
      const poly_bases base = QBBase); // Forward or Inverse

    static void execute_op(
        Context *ctx,
        uint64_t *c,
        const uint64_t *a,
        const uint64_t *b,
        const supported_operations OP,
        const poly_bases base = QBase);

    static void execute_op_by_uint(
        Context *ctx,
        uint64_t *c,
        uint64_t *a,
        uint64_t b,
        const supported_operations OP,
        const poly_bases base = QBase);

    static void execute_dualop(
        Context *ctx,
        uint64_t *c,
        uint64_t *a,
        uint64_t *b,
        uint64_t *f,
        uint64_t *d,
        uint64_t *e,
        const supported_operations OP,
        const poly_bases base = QBase);

    static void execute_seqop(
        Context *ctx,
        uint64_t *d,
        const uint64_t *a,
        const uint64_t *b,
        const uint64_t *c,
        const supported_operations OP,
        const poly_bases base = QBase);
    
};

#endif