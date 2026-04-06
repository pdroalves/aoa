#ifndef SAMPLER_H
#define SAMPLER_H

#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <newckks/cuda/htrans/common.h>
#include <newckks/cuda/htrans/ntt.h>
#include <newckks/arithmetic/poly_t.h>
#include <newckks/tool/context.h>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine

// cuRAND API errors
static const char *curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define curandCheckError(e) {                                          \
 if( e != CURAND_STATUS_SUCCESS){      \
   printf("cuRAND failure %s:%d: '%s' (%d)\n",__FILE__,__LINE__,curandGetErrorString(e), e);           \
   exit(1);                                                                 \
 }                                                                      \
}

// Discrete gaussian setup
#define HAMMINGWEIGHT (int)64
#define GAUSSIAN_STD_DEVIATION (double)3.2 //!< Standard deviation for the discrete gaussian sampling.
#define GAUSSIAN_BOUND (double)0 //!< Bound for the discrete gaussian sampling.

/**
 * @brief      A wrapper over the cuRand for sampling valid polynomials from relevant probabilistic distributions.
 * 
 * 	It follows the Singleton design pattern.
 * 	
 * @fixme Should this class really be a Singleton?
 */
class Sampler{
	private:
	curandGenerator_t gen;//!< cuRand's generator.
    curandState *states;//!< A cuRand's state for each coefficient of each polynomial residue.

	Context *ctx;
	
	double *d_tmp;
	uint64_t *h_tmp;

	public:

	bool is_init = false;

    /*! \brief Initializes Sampler.
     *
     * Allocates and initializes the cuRand objects used by Sampler's methods, which are
     * (1) a cuRand generator, and (2) a cuRand state for each coefficient of each residue.
     * 
     * @param[in] ctx The context that shall be used.
     */
	Sampler(Context *ctx);

    /*! \brief Destroy the object.
    *
    * Deallocate all related data on the device and host memory.
    */    
	~Sampler();
	
	__host__ void sample_uniform(poly_t *p, poly_bases b);

	__host__ void sample_DG(poly_t *p, poly_bases b);

	__host__ void sample_Z0(poly_t *p, poly_bases b);

	__host__ void sample_hw(poly_t *p, poly_bases b);

	__host__ void sample_narrow(poly_t *p, poly_bases b);
	
};

#endif