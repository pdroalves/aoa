#ifndef DEFINES_H
#define DEFINES_H

////////////////////////////////////////////////////////////////////////////////
// Typedefs and definitions
////////////////////////////////////////////////////////////////////////////////


#ifndef SEL
#define SEL(A, B, C) ((-(C) & ((A) ^ (B))) ^ (A))
#endif


/**
 * @brief   Supported operators by CUDAEngine::execute_polynomial_op_by_int
 */
enum add_mode_t {ADD, SUB, MUL, MULMUL, ADDADD};
/**
 * @brief      Supported RNS Bases
 */
enum poly_bases {QBase, BBase, QBBase, TBase, QTBBase};
/**
 * @brief      DGT possible directions
 */
enum dgt_direction{FORWARD, INVERSE};


/**
 * @brief  A 128 bits unsigned integer data type
 * 
 * It is compound by two 64 bits unsigned integers.
 * 
 * @param lo The lowest bits
 * @param hi The highest bits
 */
typedef struct {
  uint64_t lo = 0;
  uint64_t hi = 0;
} uint128_t;

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

/**
 * @brief The basic data type for AOADGT.
 * 
 * It provides the arithmetic required by the DGT.
 */
typedef struct GaussianInteger{
    uint64_t re; //<! The real part
    uint64_t imag;//<! The imaginary part
    inline bool operator==(const GaussianInteger& b){
        return (re == b.re) && (imag == b.imag);
    };
    __host__ __device__ inline void write(int i, uint64_t x) {
      re   = SEL(re, x, (i == 0));
      imag = SEL(imag, x, (i == 1));
    };
} GaussianInteger;


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
    GaussianInteger *d_coefs = NULL;
    /// the status of the object as an element of #poly_states,
	poly_states state = NONINITIALIZED;
    /// the base that the object lies as an element of #bases
    poly_bases base = QBase;
} typedef poly_t;

#endif