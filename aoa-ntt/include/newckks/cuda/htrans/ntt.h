#ifndef NTT_H
#define NTT_H

#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <map>
#include <sstream>      // std::ostringstream
#include <newckks/cuda/htrans/common.h>
#include <newckks/cuda/manager.h>

// Flags to switch between DGT and HDGT
#define ENABLE_HNTT_128 true
#define ENABLE_HNTT_256 true
#define ENABLE_HNTT_512 true
#define ENABLE_HNTT_1024 true
#define ENABLE_HNTT_2048 true

////////////////////////////////////////////////////////////////////////////////
// Typedefs and definitions
////////////////////////////////////////////////////////////////////////////////

// These are the factorizations used on NTT hierarchical for each N
#define NTTFACT128a (int)16
#define NTTFACT128b (int)8
#define NTTFACT256 (int)16
#define NTTFACT512a (int)32
#define NTTFACT512b (int)16
#define NTTFACT1024 (int)32
#define NTTFACT2048a (int)64
#define NTTFACT2048b (int)32
#define NTTFACT4096  (int)64
#define NTTFACT8192a (int)128
#define NTTFACT8192b (int)64
#define NTTFACT16384 (int)128
#define NTTFACT32768a (int)256
#define NTTFACT32768b (int)128
#define NTTFACT65536 (int)256
#define NTTFACT131072a (int)512
#define NTTFACT131072b (int)256
#define NTTFACT262144 (int)512
#define NTTFACT524288a (int)1024
#define NTTFACT524288b (int)512
#define NTTFACT1048576 (int)1024

/**
 * @brief      All operations related to NTT are contained in this class.
 *             It follows the Singleton design pattern.
 *             
 *             All variables are expected to be read-only after initilization.
 */
class NTTEngine{
  public:
    static bool is_init; //!< True if is initialized and ready for use. Otherwise, False.

    //////////////////
    //   Parameters //
    //////////////////
    //
    static uint64_t *d_gN; //!< Device array which stores the powers of \f$g\f$ for the second step of the hierarchical NTT.
    static uint64_t *d_ginvN; //!< Device array which stores the powers of \f$g^{-1}\f$ for the second step of the hierarchical NTT.
    static std::map<int, uint64_t*> d_nthroot; //!< Powers of n-th root of i for different signal sizes (device).
    static std::map<int, uint64_t*> d_invnthroot; //!< Powers of the inverse of n-th root of i for different signal sizes (device).
    static std::map<int, uint64_t*> h_nthroot; //!< Powers of n-th root of i for different signal sizes (host).
    static std::map<int, uint64_t*> h_invnthroot;//!< Powers of the inverse of n-th root of i for different signal sizes (host).

    /**
     * @brief Initializes NTTEngine.
     *
     * Allocates and initializes all the data structure used by NTTEngine's methods.
     * Pre-computes g^j and its modular inverse.
     */
    static void init();
  
    /*! \brief Destroy the object.
    *
    * Deallocate all related data on the device and host memory.
    */    
    static void destroy();

    static void execute_ntt(
      Context *ctx,
      poly_t *data,
      const transform_directions direction); // Forward or Inverse

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
        const uint64_t *a,
        const uint64_t b,
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
        uint64_t *a,
        uint64_t *b,
        uint64_t *c,
        const supported_operations OP,
        const poly_bases base = QBase);
    
};

#endif