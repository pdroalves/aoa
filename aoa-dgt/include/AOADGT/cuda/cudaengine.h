#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdexcept>
#include <cstring>
#include <omp.h>
#include <NTL/RR.h>
#include <iomanip>
#include <NTL/ZZ.h>
#include <AOADGT/settings.h>
#include <AOADGT/arithmetic/polynomial.h>
#include <AOADGT/arithmetic/context.h>
#include <AOADGT/cuda/dgt.h>

using namespace NTL;

/**
 * \brief Parameters to initialize CUDAEngine
 */
typedef struct{
  /// The size of the main base, usually referred as Q.
  int k; 
  /// The size of the auxiliar base, usually referred as B. Equal to k+1 by default.
  int kl = -1; 
  /// The degree of the cyclotomic ring.
  int nphi; 
  /// Defines the plaintext domain for the BFV, or the scaling factor used in the CKKS.
  uint64_t pt; 
} CUDAParams;

// Defined in polynomial.h
size_t poly_get_size(poly_bases base);

class Context;

/**
 * @brief  All generic operations that shall be executed on the GPGPU
 *         are contained in this class. It follows the Singleton design pattern.
 *
 *         All variables are expected to be read-only after initialization.
 *
 *         \todo Refactor to support multiple t's.
 *         
 *         
 */
class CUDAEngine{
  public:

    //////////////////
    //   Parameters //
    //////////////////
    static int is_init; //!< True if CUDAEngine is initialized and ready for use. Otherwise, False.
    
    /*! \brief Defines that all computation will be made in a 2*N-degree cyclotomic ring
    * 
    * The cyclotomic polynomial used by AOADGT is unique and defined by 2*N. 
    */
    static int N;
    static uint64_t t; //!< Defines the plaintext domain

    static int scalingfactor; //!< Stores the precision requested during initialization (CKKS only!)
    static int dnum; //!< To be used on the implementation of CKKS' bootstrap (CKKS only!)

    /////////
    // RNS //
    /////////
    static uint64_t *RNSCoprimes; //!< A pointer to the constant memory array.

    static std::vector<uint64_t> RNSPrimes;//!< The main basis used by RNS.
    static std::vector<uint64_t> RNSBPrimes;//!< The secondary basis used by RNS.
    static ZZ RNSProduct; //!< The product of all coprimes used in the main basis of RNS.
    static ZZ RNSBProduct;//!< The product of all coprimes used in the secondary basis of RNS.
    static std::vector<ZZ> RNSMpi;//!< The i-nth element stores \f$ \frac{RNSProduct}{RNSPrimes[i]} \pmod {RNSPrimes[i]}\f$.
    static std::vector<uint64_t> RNSInvMpi;//!< The i-nth element stores \f$\frac{RNSProduct}{RNSPrimes[i]}^{-1} \pmod {RNSPrimes[i]} \f$.


    ////////////////////
    // Init / Destroy //
    ////////////////////
    /*! \brief Initializes CUDAEngine for execution with a single t-value.
     *
     * Allocates and initializes all the data structure used by CUDAEngine's methods.
     * Most of them are RNS related arrays. At the end, calls DGTEngine::init().
     * 
     * @param[in] k  Size of the main RNS basis.
     * @param[in] kl Size of the secondary RNS basis.
     * @param[in] M  Defines the M-th  cyclotomic polynomial to be used by AOADGT.
     * @param[in] t  Defines the plaintext domain \f$R_t\f$ for BFV, or the scaling factor of the CKKS.
     */
    static void init(
        const int k,
        const int kl,
        const int M,
        const uint64_t t);

    /*! \brief Initializes CUDAEngine for execution with a single t-value.
     *
     * Allocates and initializes all the data structure used by CUDAEngine's methods.
     * Most of them are RNS related arrays. At the end, calls DGTEngine::init().
     * 
     * 
     * @param[in] p A CUDAParams object
     */
    static void init(CUDAParams p);
  
    /*! \brief Destroy the object.
    *
    * Deallocate all related data on the device and host memory.
    */    
    static void destroy();

    /*! \brief Compute all values required to generate the RNS basis.
     * 
     * Extract coprimes from the hardcoded list and copy to the GPU memory.
     * 
     * @param[in] k      Size of q-basis
     * @param[in] kl     Size of b-basis
     */
    static void gen_rns_primes(
        unsigned int k,
        unsigned int kl);


    /*! \brief Precompute all values used by CUDAEngine.
     * 
     * Compute the intermediary values required by RNS, the HPS method, and the cryptosystems..
     * 
     */
    static void precompute();

    /*! \brief Negate all coefficients.
    *
    * Compute \f$b = -a \pmod R_q\f$
    *  
    * @param[out] b     Outcome
    * @param[in]  a     The input
    * @param[in]  base  The input base
    * @param[in]  ctx   The Context
    */
    static void execute_polynomial_negate(
      GaussianInteger *b,
      const GaussianInteger *a,
      const int base,
      Context *ctx);

    /*! \brief Apply a basic polynomial operation against an integer.
     *
     * Computes \f$c = a \diamond b \pmod R_q\f$ for some operator \f$\diamond\f$, 
     * for the supported operators described in #add_mode_t.
     *
     * @param[out] b         Outcome
     * @param[in]  a         a polynomial
     * @param[in]  base      base
     * @param[in]  x         an integer
     * @param      OP    An element of #add_mode_t
     * @param      ctx       the context
     */
    static void execute_polynomial_op_by_int(
      GaussianInteger *b,
      GaussianInteger *a,
      uint64_t x,
      const int base,
      add_mode_t OP,
      Context *ctx);


    /**
     * @brief      Applies two operators involving polynomials and integers in a single CUDA kernel
     *
     * Computes: \f$ b1 = a1 \diamond a1\f$ and \f$b2 = a2 \diamond a2\f$, 
     * for the supported operators described in #add_mode_t.
     * 
     * @param[out]     b1    The b 1
     * @param[in]      a1    A 1
     * @param[out]     b2    The b 2
     * @param[in]      a2    A 2
     * @param         base  The base
     * @param[in]     x1    The x 1
     * @param[in]     x2    The x 2
     * @param         OP    An element of #add_mode_t
     * @param         ctx   The context
     */
    static void execute_polynomial_double_op_by_int(
    GaussianInteger *b1,
    GaussianInteger *a1,
    GaussianInteger *b2,
    GaussianInteger *a2,
    uint64_t x1,
    uint64_t x2,
    add_mode_t OP,
    const poly_bases base,
    Context *ctx );

    /*! \brief The HPS method for scaling by \f$\frac{t}{q}\f$.
    *
    * Executes the simple scaling procedure described in section 2.3 
    * but as a piece of the complex scaling procedure. So that,
    *
    * \f$b = a\frac{tp}{qp} \pmod b\f$
    *
    * "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[out] b      Outcome in base t
    * @param[in]  a_Q    Input in base q
    * @param[in]  a_B    Input in base b
    * @param[in]  ctx    the context
    */
    static void execute_polynomial_scaling_tDivQ_mod_p(
      uint64_t *b,
      const GaussianInteger *a_Q,
      const GaussianInteger *a_B,
      Context *ctx);

    /*! \brief The HPS method for scaling by \f$\frac{t}{q} \pmod t\f$.
    *
    * Executes the simple scaling procedure inplace, as described in section 2.3 so that
    * \f$b = a\frac{t}{q} \pmod t\f$
    *
    * "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[in,out] a      Outcome in base q
    * @param[in]     ctx    the context
    */
    static void execute_polynomial_simple_scaling(
      GaussianInteger *a,
      Context *ctx);


    /*! \brief The HPS method for scaling by \f$\frac{t}{q}\f$.
    *
    * Executes the complex scaling procedure described in section 2.4
    *
    * \f$b = a\frac{tp}{qp} \pmod b\f$
    *
    * "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[out] b      Outcome in base t
    * @param[in]  a_Q    Input in base q
    * @param[in]  a_B    Input in base b
    * @param[in]  ctx    the context
    */
    static void execute_polynomial_complex_scaling(
      uint64_t *b,
      const GaussianInteger *a_Q,
      const GaussianInteger *a_B,
      Context *ctx);
    
    /*! \brief The HPS method for computing basis B from residues on a base Q.
    *
    * "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[in,out] a      array in base q
    * @param[in]     ctx    the context
    */
    static  void execute_polynomial_basis_ext_Q_to_B(
      GaussianInteger *a,
      Context *ctx);

    /*! \brief The HPS method for computing basis B from residues on a base Q.
    *
    * Todo: Refactor using templates
     * 
    * "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[out] b      Outcome in base q
    * @param[in]  a      Input in base b
    * @param[in]  ctx    the context
    */
    static  void execute_polynomial_basis_ext_B_to_Q(
      GaussianInteger *b,
      uint64_t *a,
      Context *ctx);
    /**
     * @brief      Receives an polynomial represented in base QB and computed the extension in base Q.
     *
     *            This method works out of place.
     *
     * @param      ctx    The context
     * @param      a      The input in base QB
     * @param      b      The output in base B
     * @param[in]  level  The level
     */
    static void execute_approx_modulus_reduction(
      Context *ctx,
      GaussianInteger *a,
      GaussianInteger *b,
      int level);
    /**
     * @brief      Receives an polynomial represented in base Q and computed the extension in base B.
     *
     *            This method works in place.
     *
     * @param      ctx    The context
     * @param      a      The polynomial (and the outcome)
     * @param[in]  level  The level
     */
    static void execute_approx_modulus_raising(
      Context *ctx,
      GaussianInteger *a,
      int level);


    /*! \brief The HPS method for computing \f$xi\f$, used by FV's homomorphic multiplication
    *
    * Seacon 4.5 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[out] b      Outcome of many polynomials in base q
    * @param[in]  a      Input in base q
    * @param[in]  ctx    the context
    */
    static void execute_xi_rns(
      GaussianInteger **b,
      GaussianInteger *a,
      Context *ctx);
    
    /*! \brief The HPS method for computing \f$rho\f$, used by FV's keygen
    *
    * Seacon 4.5 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[out] b      Outcome of many polynomials in base q
    * @param[in]  a      Input in base q
    * @param[in]  ctx    the context
    */
    static void execute_rho_bfv_rns(
      GaussianInteger **b,
      GaussianInteger *a,
      Context *ctx);


    /*! \brief The HPS method for computing \f$rho\f$, used by CKKS's keygen
    *
    * Seacon 4.5 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[out] b      Outcome of many polynomials in base q
    * @param[in]  a      Input in base q
    * @param[in]  ctx    the context
    */
    static void execute_rho_ckks_rns(
      GaussianInteger *b,
      GaussianInteger *a,
      Context *ctx);
    
    /*! \brief The HPS method for computing \f$xi\f$, used by CKKS's keygen
    *
    * Seacon 4.5 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
    * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
    *
    * @param[out] b      Outcome of many polynomials in base q
    * @param[in]  a      Input in base q
    * @param[in]  ctx    the context
    */
    static void execute_xi_ckks_rns(
      GaussianInteger *b,
      GaussianInteger *a,
      Context *ctx);


    /**
     * @brief      Computes the rescale procedure used by CKKS
     *
     * @param[in]      a      The input
     * @param[out]      b      The rescaled output
     * @param     level  The level
     * @param     ctx    The context
     */
    static void execute_ckks_rescale(
      GaussianInteger *a,
      GaussianInteger *b,
      const int level,
      Context *ctx);

    /**
     * @brief      Return the quantity of residues for a certain base
     *
     * @param[in]  base  Defines the basis
     *
     * @return     The quantity of residues for that basis.
     */
    static int get_n_residues(int base);
};

#endif
