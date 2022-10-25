#ifndef COPRIMES_H
#define COPRIMES_H

#include <map>
#include <newckks/cuda/htrans/common.h>

#define MAX_COPRIMES 500 //!< The maximum quantity of coprimes that may be used
#define MAX_COPRIMES_IN_A_BASE 100 //!< The maximum quantity of coprimes that may be used in a single base

/**
 * @brief       A map of primitive roots
 * 
 * Stores primitive roots for different coprimes.
 */
extern std::map<uint64_t, int> PROOTS;
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

/**
 * @brief       A map of n-th roots.
 * 
 * Stores n-th roots of i for different coprimes.
 */
extern std::map<uint64_t, std::map<int, GaussianInteger>> GI_NTHROOT;

/**
 * @brief       A map of inverses of n-th roots.
 * 
 * Stores inverses of n-th roots of i for different coprimes.
 */
extern std::map<uint64_t, std::map<int, GaussianInteger>> GI_INVNTHROOT;
#endif