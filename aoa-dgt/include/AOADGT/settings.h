#ifndef SETTINGS_H
#define SETTINGS_H
#include <stdio.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <time.h>
#include <map>
#include <AOADGT/defines.h>
#include <AOADGT/tool/version.h>
#include "AOADGTConfig.h"

#define HAMMINGWEIGHT_H 64

/**
 * @brief       A map of n-th roots.
 * 
 * Stores n-th roots of i for different coprimes.
 */
extern std::map<uint64_t, std::map<int, GaussianInteger>> NTHROOT;

/**
 * @brief       A map of inverses of n-th roots.
 * 
 * Stores inverses of n-th roots of i for different coprimes.
 */
extern std::map<uint64_t, std::map<int, GaussianInteger>> INVNTHROOT;

/**
 * @brief       A map of primitive roots
 * 
 * Stores primitive roots for different coprimes.
 */
extern std::map<uint64_t, int> PROOTS;

// default block size
#define ADDBLOCKXDIM 512 //!< A suggestion for a default CUDA block size. @fixme We should avoid this.
#define FPERROR 1e-9 //!< A tolerance epsilon for HPS floating-point division.

// RNS
#define MAX_COPRIMES 200 //!< The maximum quantity of coprimes that may be used in total.
#define MAX_COPRIMES_IN_A_BASE MAX_COPRIMES/2  //!< The maximum quantity of coprimes that may be used in a single base.

//
extern uint64_t COPRIMES_45_BUCKET[]; //!< A list of 45-bits coprimes supported for RNS.
extern uint64_t COPRIMES_48_BUCKET[]; //!< A list of 48-bits coprimes supported for RNS.
extern uint64_t COPRIMES_50_BUCKET[]; //!< A list of 50-bits coprimes supported for RNS.
extern uint64_t COPRIMES_52_BUCKET[]; //!< A list of 52-bits coprimes supported for RNS.
extern uint64_t COPRIMES_55_BUCKET[]; //!< A list of 55-bits coprimes supported for RNS.
extern uint64_t COPRIMES_62_BUCKET[]; //!< A list of 62-bits coprimes supported for RNS.
extern uint64_t COPRIMES_63_BUCKET[]; //!< A list of 63-bits coprimes supported for RNS.

extern const uint32_t COPRIMES_45_BUCKET_SIZE; //!< The size of COPRIMES_45_BUCKET.
extern const uint32_t COPRIMES_48_BUCKET_SIZE; //!< The size of COPRIMES_48_BUCKET.
extern const uint32_t COPRIMES_50_BUCKET_SIZE; //!< The size of COPRIMES_50_BUCKET.
extern const uint32_t COPRIMES_52_BUCKET_SIZE; //!< The size of COPRIMES_52_BUCKET.
extern const uint32_t COPRIMES_55_BUCKET_SIZE; //!< The size of COPRIMES_55_BUCKET.
extern const uint32_t COPRIMES_62_BUCKET_SIZE; //!< The size of COPRIMES_62_BUCKET.
extern const uint32_t COPRIMES_63_BUCKET_SIZE; //!< The size of COPRIMES_63_BUCKET.

extern uint64_t COPRIMES_BUCKET[]; //!< A list of coprimes supported for RNS used during the execution.
extern uint32_t COPRIMES_BUCKET_SIZE; //!< The size of COPRIMES_BUCKET.

// Auxiliar methods
extern double compute_time_ms(struct timespec start,struct timespec stop);
extern uint64_t get_cycles();

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

// Discrete gaussian setup
#define GAUSSIAN_STD_DEVIATION (float)3.2 //!< Standard deviation for the discrete gaussian sampling.
#define GAUSSIAN_BOUND (float)0 //!< Bound for the discrete gaussian sampling.

#endif
