#ifndef DGT_H
#define DGT_H

#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <map>
#include <NTL/ZZ.h>
#include <AOADGT/settings.h>
#include <AOADGT/cuda/cudaengine.h>
#include <AOADGT/defines.h>
#include <cuda_runtime.h>
#include <sstream>      // std::ostringstream

class Context;

////////////////////////////////////////////////////////////////////////////////
// Typedefs and definitions
////////////////////////////////////////////////////////////////////////////////
#define WORDSIZE (int)64

#define ENABLE_HDGT_1024 true
#define ENABLE_HDGT_2048 true

// These are the factorizations used on DGT hierarchical for each N
#define DGTFACT1024 (int)32
#define DGTFACT1024 (int)32
#define DGTFACT2048a (int)64
#define DGTFACT2048b (int)32
#define DGTFACT4096  (int)64
#define DGTFACT8192a (int)128
#define DGTFACT8192b (int)64
#define DGTFACT16384 (int)128
#define DGTFACT32768a (int)256
#define DGTFACT32768b (int)128
////////////////////////////////////////////////////////////////////////////////
// Methods
////////////////////////////////////////////////////////////////////////////////
__host__ __device__  uint128_t mullazy(const uint64_t a, const uint64_t b);
__host__ __device__ uint64_t mulmod(uint64_t a, uint64_t b, int rid); //!<  Modular multiplication in Z_{d_RNSCoprimes[rid]}
__device__ void mulint_dgt(GaussianInteger *c, const GaussianInteger a, const uint64_t b, const int rid); //!<  Modular multiplication by a int in Z_{d_RNSCoprimes[rid]}
__device__ GaussianInteger mulint_dgt(const GaussianInteger a, const uint64_t b, const int rid); //!<  Modular multiplication by a int in Z_{d_RNSCoprimes[rid]}
__host__ __device__ uint64_t fast_pow(uint64_t a, uint64_t b, int rid); //!<  Modular exponentiation in Z_{d_RNSCoprimes[rid]}
__host__ __device__ uint64_t mod(uint128_t a, int rid);//!<  Modular reduction by Z_{d_RNSCoprimes[rid]}
__host__ __device__ GaussianInteger GIAdd(GaussianInteger a, GaussianInteger b, int rid); //!<  Modular addition in GF({d_RNSCoprimes[rid]}^2)
__host__ __device__ GaussianInteger GISub(GaussianInteger a, GaussianInteger b, int rid); //!<  Modular subtraction in GF({d_RNSCoprimes[rid]}^2)
__host__ __device__ GaussianInteger GIMul(GaussianInteger a, GaussianInteger b, int rid); //!<  Modular multiplication in GF({d_RNSCoprimes[rid]}^2)
__host__ __device__ GaussianInteger GIMul(GaussianInteger a, uint64_t b, int rid); //!<  Modular multiplication by a int in GF({d_RNSCoprimes[rid]}^2)
__host__ __device__ void GIAdd(GaussianInteger *c, GaussianInteger a, GaussianInteger b, int rid); //!<  Modular addition in GF({d_RNSCoprimes[rid]}^2)
__host__ __device__ void GISub(GaussianInteger *c, GaussianInteger a, GaussianInteger b, int rid); //!<  Modular subtraction in GF({d_RNSCoprimes[rid]}^2)
__host__ __device__ void GIMul(GaussianInteger *c, GaussianInteger a, uint64_t b, int rid); //!<  Modular multiplication by a int in GF({d_RNSCoprimes[rid]}^2)

/**
 * @brief       Returns A if C == 0 and B if C == 1
 *
 * @param      A     { parameter_description }
 * @param      B     { parameter_description }
 * @param      C     { parameter_description }
 *
 * @return     { description_of_the_return_value }
 */
#define SEL(A, B, C) ((-(C) & ((A) ^ (B))) ^ (A))
#ifdef __CUDA_ARCH__
extern __constant__ uint64_t d_RNSCoprimes[];
#define addmod(a, b, rid) SEL(a+b,a+b-d_RNSCoprimes[rid],(a+b) >= d_RNSCoprimes[rid])
#define submod(a, b, rid) SEL(a-b,a+(d_RNSCoprimes[rid]-b),a<b)
#else
#define addmod(a, b, rid) SEL(a+b,a+b-COPRIMES_BUCKET[rid],(a+b) >= COPRIMES_BUCKET[rid])
#define submod(a, b, rid) SEL(a-b,a+(COPRIMES_BUCKET[rid]-b),a<b)
#endif
#define submodp(a, b, p) SEL(a-b,a+(p-b),a<b)

/**
 * @brief      All operations related to DGT are contained in this class.
 *             It follows the Singleton design pattern.
 *             
 *             All variables are expected to be read-only after initilization.
 */
class DGTEngine{
  public:

    //////////////////
    //   Parameters //
    //////////////////
    //
    static bool is_init; //!< True if DGTEngine is initialized and ready for use. Otherwise, False.
    static std::map<int, uint64_t*> d_gjk; //!< Device array which stores the powers of \f$g\f$ for the DGT.
    static std::map<int, uint64_t*> d_invgjk; //!< Device array which stores the powers of \f$g^{-1}\f$ for the DGT.
    static uint64_t *d_gN; //!< Device array which stores the powers of \f$g\f$ for the second step of the hierarchical DGT.
    static uint64_t *d_ginvN; //!< Device array which stores the powers of \f$g^{-1}\f$ for the second step of the hierarchical DGT.
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

    /**
     * @brief      Apply the DGT.
     *
     * Applies the DGT on a polynomial following mainly the algorithm proposed 
     * by Badawi and Star at "Efficient Polynomial Multiplication via Modified 
     * Discrete Galois Transform and Negacyclic Convolution". 
     * 
     * When \f$CUDAEngine::N \leq 512\f$ their standard algorithm is executed
     * on using a single CUDA kernel.
     * 
     * For higher degrees we recall the Hierarchical FFT algorithm, discussed
     * by Govindaraju et al. at "High performance discrete Fourier transforms 
     * on graphics processors", and break the polynomial on smaller blocks.
     * For this case we require 4 CUDA kernels.
     * 
     * @param[in,out]      data       A pointer to the d_coefs array which shall receive the transformation.
     * @param[in]  base       The base
     * @param[in]  direction  Specify the DGT direction: FORWARD or INVERSE
     * @param[in]  ctx        The context
     */
    static void execute_dgt( Context *ctx, poly_t* p, const dgt_direction direction );

    /*! \brief Compute a polynomial addition.
     *
     * Computes \f$c = a + b\f$
     *
     * @param[out] c         Outcome
     * @param[in]  a         First operator
     * @param[in]  b         Second operator
     * @param[in]  base      The RNS basis which operators are.
     * @param[in]  stream    The cudaStream_t that shall be used.
     */
    static void execute_add_dgt(
      GaussianInteger *c,
      const GaussianInteger *a,
      const GaussianInteger *b,
      const int base,
      const cudaStream_t stream
    );

    /*! \brief Compute a polynomial subtraction.
     *
     * Computes \f$c = a - b\f$
     *
     * @param[out] c         Outcome
     * @param[in]  a         First operator
     * @param[in]  b         Second operator
     * @param[in]  base      The RNS basis which operators are.
     * @param[in]  stream    The cudaStream_t that shall be used.
     */
    static void execute_sub_dgt(
      GaussianInteger *c,
      const GaussianInteger *a,
      const GaussianInteger *b,
      const int base,
      const cudaStream_t stream
    );

    /*! \brief Compute two polynomial additions.
     *
     * Computes \f$c_1 = a_1 + b_1\f$ and \f$c_2 = a_2 + b_2\f$
     *
     * @param[out] c1         Outcome of the first addition
     * @param[in]  a1         First operator of the first addition
     * @param[in]  b1         Second operator of the first addition
     * @param[out] c2         Outcome of the second addition
     * @param[in]  a2         First operator of the second addition
     * @param[in]  b2         Second operator of the second addition
     * @param[in]  base      The RNS basis which operators are.
     * @param[in]  stream    The cudaStream_t that shall be used.
     */
    static void execute_double_add_dgt(
      GaussianInteger *c1,
      const GaussianInteger *a1,
      const GaussianInteger *b1,
      GaussianInteger *c2,
      const GaussianInteger *a2,
      const GaussianInteger *b2,
      const int base,
      const cudaStream_t stream
    );


    /**
     * @brief      Computes the DR2 operation used by BFV's and CKKS' homomorphic multiplication
     *
     * @param[out]      ct21    The ct21
     * @param[out]      ct22    The ct22
     * @param[out]      ct23    The ct23
     * @param[in]  ct01    The ct01
     * @param[in]  ct02    The ct02
     * @param[in]  ct11    The ct11
     * @param[in]  ct12    The ct12
     * @param[in]  base    The base
     * @param[in]  stream  The stream
     */
    static void execute_dr2_dgt(
    GaussianInteger *ct21, // Outcome
    GaussianInteger *ct22, // Outcome
    GaussianInteger *ct23, // Outcome
    const GaussianInteger *ct01, // Operand 1
    const GaussianInteger *ct02, // Operand 1
    const GaussianInteger *ct11, // Operand 2
    const GaussianInteger *ct12, // Operand 2
    const int base,
    const cudaStream_t stream);

    /*! \brief Compute a polynomial multiplication.
     *
     * Computes \f$c = a \times b\f$. 
     * This method is a template that supports the multiplication between poly_t s
     * (with GaussianInteger coefficients) and between a poly_t and a uint64_t
     * element.
     *
     * @param[out] c         Outcome
     * @param[in]  a         First operator
     * @param[in]  b         Second operator
     * @param[in]  base      The RNS basis which operators are.
     * @param[in]  stream    The cudaStream_t that shall be used.
     */
    static void execute_mul_dgt_gi(
      GaussianInteger *c,
      const GaussianInteger *a,
      const GaussianInteger *b,
      const int base,
      const cudaStream_t stream
    );

    static void execute_mul_dgt_u64(
      GaussianInteger *c,
      const GaussianInteger *a,
      const uint64_t *b,
      const int base,
      const cudaStream_t stream
    );


    /*! \brief Compute a polynomial multiplication followed by a polynomial addition.
     *
     * Computes \f$d = (a \times b) + c\f$ in a single CUDA kernel.
     *
     * @param[out] d         Outcome
     * @param[in]  a         First operator
     * @param[in]  b         Second operator
     * @param[in]  c         Third operator
     * @param[in]  base      The RNS basis which operators are.
     * @param[in]  stream    The cudaStream_t that shall be used.
     */
    static void execute_muladd_dgt(
      GaussianInteger *d,
      const GaussianInteger *a,
      const GaussianInteger *b,
      const GaussianInteger *c,
      const int base,
      const cudaStream_t stream
    );
};

#endif