#ifndef DEFINES_H
#define DEFINES_H

enum transform_directions {
	FORWARD,
	INVERSE
};

enum engine_types {
  NTTTrans,
  DGTTrans
};

enum supported_operations {
    ADDOP, ///< Addition
    SUBOP, ///< Subtraction
    MULOP, ///< Multiplication
    NEGATEOP, ///< Negate
    ADDADDOP, ///< Two non-related additions
    MULandADDOP ///< Multiplication followed by an addition on the same operands
};

/**
 * @brief  A 128 bits unsigned integer data type
 * 
 * It is compound by two 64 bits unsigned integers.
 * 
 * @param lo The lowest bits
 * @param hi The highest bits
 */
struct u128{
  uint64_t lo = 0;
  uint64_t hi = 0;
  __host__ __device__ u128(uint64_t x = 0){
    lo = x;
  };
} typedef uint128_t;

/**
 * @brief      Defines the states supported for a poly_t object
 * 
 * There are three possible states.
 */
enum poly_states {
  NONINITIALIZED,///< This object cannot be used in this state
  RNSSTATE,///< Residues are in the RNS domain
  TRANSSTATE///< Residues are in the transform domain
};

enum poly_bases {
  QBase,
  BBase,
  QBBase
};

/**
 * @brief      Defines a polynomial.
 * 
 * It contains:
 * - a boolean indicating if the object was initialized,
 * - a vector of coefficients on the host,
 * - an array of residues on the device,
 * - the status of the object (related to poly_states),
 * - the base that the object lies in (related to settings.h bases enum_t).
 */
struct polynomial {
    /// an array of residues on the device,
    uint64_t *d_coefs = NULL;
    /// the status of the object as an element of #poly_states,
    poly_states state = NONINITIALIZED;
    poly_bases base;
} typedef poly_t;

/**
 * @brief The basic data type for the DGT.
 * 
 */
typedef struct GaussianInteger{
    uint64_t re; //<! The real part
    uint64_t imag;//<! The imaginary part
} GaussianInteger;

// Secret Key type
typedef struct{
    poly_t s;
} SecretKey;

// swk type
typedef struct{
    poly_t a;
    poly_t b;
} SwitchKey;

// Public Key type
typedef SwitchKey PublicKey;
// evk type
typedef SwitchKey EvaluationKey;
// rtk type
typedef SwitchKey RotationKey;
// cjk type
typedef SwitchKey ConjugationKey;

#endif