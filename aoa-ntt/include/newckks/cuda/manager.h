#ifndef CUDA_MANAGER_H
#define CUDA_MANAGER_H

#include <newckks/cuda/htrans/common.h>
#include <newckks/tool/context.h>
#include <newckks/arithmetic/poly_t.h>
#include <newckks/cuda/htrans/common.h>
#include <newckks/cuda/htrans/ntt.h>
#include <newckks/cuda/htrans/dgt.h>
#include <NTL/ZZ.h>

#define DEFAULTBLOCKSIZE 512
#define get_grid_dim(size, blocksize) (size%blocksize == 0? size/blocksize : size/blocksize + 1)
#define is_power2(t) t && !(t & (t - 1))

class CUDAManager{
	public:
    //////////////////
    //   Parameters //
    //////////////////
    static int is_init; //!< True if CUDAEngine is initialized and ready for use. Otherwise, False.
    
    /*! \brief Defines that all computation will be made in a 2*N-degree cyclotomic ring
    * 
    * The cyclotomic polynomial used by cuPoly is unique and defined by 2*N. 
    */
    static int N;
    static int dnum; //!< Decomposition factor
    static uint64_t scalingfactor;
    /////////
    // RNS //
    /////////
    static uint64_t *RNSCoprimes; //!< A pointer to the constant memory array.

    static std::vector<uint64_t> RNSQPrimes;//!< The main basis used by RNS.
    static std::vector<uint64_t> RNSBPrimes;//!< The secondary basis used by RNS.


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
     * @param[in] N  Defines the M-th  cyclotomic polynomial to be used by cuPoly.
     * @param[in] t  Defines the plaintext domain \f$R_t\f$ for BFV, or the scaling factor of the CKKS.
     */
    static void init(
        const int k,
        const int kl,
        const int N,
        const uint64_t scalingfactor,
        engine_types e = NTTTrans);

    /*! \brief Initializes CUDAEngine for execution with a single t-value.
     *
     * Allocates and initializes all the data structure used by CUDAEngine's methods.
     * Most of them are RNS related arrays. At the end, calls DGTEngine::init(). 
     * 
     * This version selects automatically the best size for the secondary RNS basis.
     * 
     * @param[in] k  Size of the main RNS basis.
     * @param[in] N  Defines the M-th  cyclotomic polynomial to be used by cuPoly.
     * @param[in] t  Defines the plaintext domain \f$R_t\f$ for BFV, or the scaling factor of the CKKS.
     */
    static void init(
        const int k,
        const int N,
        const uint64_t scalingfactor,
        engine_types e = NTTTrans);

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

    static NTL::ZZ get_q_product(){
        NTL::ZZ M = NTL::to_ZZ(1);
        for(auto x : RNSQPrimes)
            M = x * M;
        return M;
    }

    static NTL::ZZ get_b_product(){
        NTL::ZZ M = NTL::to_ZZ(1);
        for(auto x : RNSBPrimes)
            M = x * M;
        return M;
    }

    static int get_n_residues(poly_bases b){
        switch(b){
            case QBase:
            return RNSQPrimes.size();
            break;
            case BBase:
            return RNSBPrimes.size();
            break;
            default:
            return RNSQPrimes.size() + RNSBPrimes.size();
        }
    };


    /*! \brief Precompute all values used by CUDAEngine.
     * 
     * Compute the intermediary values required by RNS, the HPS method, and the cryptosystems..
     * 
     */
    static void precompute();

    static void execute_dot(
        Context *ctx,
        uint64_t *c,
        uint64_t **a,
        uint64_t **b,
        const int k);

    static void execute_rho(
        Context *ctx,
        uint64_t *b,
        uint64_t *a);

    static void execute_rescale(
        Context *ctx,
        uint64_t *a,
        uint64_t *b,
        const int level);

    static void execute_modup(
        Context *ctx,
        uint64_t *a,
        uint64_t *b,
        const int level);

    static void execute_moddown(
        Context *ctx,
        uint64_t *a,
        uint64_t *b,
        const int level);

    static void execute_dr2(
        Context *ctx,
        uint64_t *ct21, // Outcome
        uint64_t *ct22, // Outcome
        uint64_t *ct23, // Outcome
        const uint64_t *ct01, // Operand 1
        const uint64_t *ct02, // Operand 1
        const uint64_t *ct11, // Operand 2
        const uint64_t *ct12); // Operand 2
};
#endif