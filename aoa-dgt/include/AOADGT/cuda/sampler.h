#ifndef SAMPLER_H
#define SAMPLER_H

#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <AOADGT/settings.h>
#include <AOADGT/arithmetic/polynomial.h>
#include <omp.h>

#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine

enum pdistributions
{
	DISCRETE_GAUSSIAN, ///< Discrete Gaussian
	HAMMINGWEIGHT, ///< Binary with Hamming weight #HAMMINGWEIGHT_H
	NARROW, ///< Samples from {0, 1, 2} 
	ZO, ///< 
	BINARY, ///< Binary
	UNIFORM, ///< Uniform in R_q
	KINDS_COUNT ///< Just a counter
};

/// \todo Randomize the SEED
#define SEED (unsigned long long)(42)// We derandomize for debug purposes

/**
 * @brief      A wrapper over the cuRand for sampling valid polynomials from relevant probabilistic distributions.
 * 
 * 	It follows the Singleton design pattern.
 * 	
 * @fixme Should this class really be a Singleton?
 */
class Sampler{
	public:
	static curandGenerator_t gen;//!< cuRand's generator.
	static curandState *states;//!< A cuRand's state for each coefficient of each polynomial residue.

    /*! \brief Initializes Sampler.
     *
     * Allocates and initializes the cuRand objects used by Sampler's methods, which are
     * (1) a cuRand generator, and (2) a cuRand state for each coefficient of each residue.
     * 
     * @param[in] ctx The context that shall be used.
     */
	static void init(Context *ctx);
	
	static void reset(Context *ctx);

	/**
	 * @brief Sample from a probabilistic distribution.
	 * 
	 * @param[in]	ctx		The context
	 * @param[out]	p		The poly_t object that shall receive the outcome
	 * @param[in]	kind  	Defines the probabilistic distribution that shall be used. References to #pdistributions.
	 */
	static void sample(Context *ctx, poly_t *p, int kind);

	/**
	 * @brief Sample from a probabilistic distribution.
	 * 
	 * Instantiates internally a poly_t object.
	 * 
	 * @param[in]	ctx		The context
	 * @param[in]	kind  	Defines the probabilistic distribution that shall be used. References to #pdistributions.
	 * @return				A poly_t object that shall receive the outcome
	 */
	static poly_t* sample(Context *ctx, int kind);

    /*! \brief Destroy the object.
    *
    * Deallocate all related data on the device and host memory.
    */    
	static void destroy(){
		curandDestroyGenerator(gen);
		cudaCheckError();
		
		cudaFree(states);
		cudaCheckError();
	}
	
};
#endif
