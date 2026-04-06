#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <vector>
#include <NTL/ZZ.h>
#include <map>
#include <algorithm>
#include <sstream>
#include <AOADGT/settings.h>
#include <AOADGT/cuda/cudaengine.h>
#include <AOADGT/arithmetic/context.h>
#include <AOADGT/tool/log.h>
#include <AOADGT/cuda/dgt.h>
#include <omp.h>

NTL_CLIENT

/**
 * @brief      Return the size in bytes required to store
 *             all the residues of a certain base
 *             
 * @param base the base
 *
 * @return     the size in bytes required to store all the residues of a certain base
 */
__host__ size_t poly_get_size(poly_bases base);

/**
 * @brief Initializes a poly_t object.
 * 
 * @param[in] ctx the context
 * @param[in, out] a The object
 * @param[in] base the base (the default value is QBase)
 */
__host__ void poly_init(Context *ctx, poly_t *a, poly_bases base = QBase);

/**
 * @brief Releases the memory related to a poly_t object
 * 
 * @param[in] a the object
 */
__host__ void poly_free(Context *ctx, poly_t *a);

/**
 * @brief Clear the memory related to a poly_t object without deallocating it.
 * 
 * Write 0's to all arrays.
 * 
 * @param ctx The context
 * @param a The object
 * 
 */
__host__ void poly_clear(Context *ctx, poly_t *a);


__host__ void poly_copy_to_device(Context *ctx, poly_t *a, uint64_t *h_coefs);
__host__ uint64_t* poly_copy_to_host(Context *ctx, poly_t *a);

/**
 * @brief       Copy all coefficients from "a" to "b"
 * 
 * This method copy coefficient privileging the RNSSTATE, what means that if
 * the object status is RNSSTATE or BOTHSTATE it will execute the copy within device's
 * memory and change the status to RNSSTATE. Otherwise it will execute the copy on the 
 * host by **will not** call poly_copy_to_device().
 * 
 * @param ctx   The context
 * @param b     Destiny
 * @param a     Source
 */
__host__ void poly_copy( Context *ctx,  poly_t *b, poly_t *a);


/**
 * @brief       Returns the polynomial degree.
 * 
 * @param ctx   The context
 * @param  a    the object
 */
__host__ int poly_get_deg(Context *ctx, poly_t *a);

/**
 * @brief       Executes a polynomial addition.
 * 
 * Computes \f$c = a + b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
__host__ void poly_add(Context *ctx, poly_t *c, poly_t *a, poly_t *b);

/**
 * @brief Compute two polynomial additions.
 *
 * Computes \f$c_1 = a_1 + b_1\f$ and \f$c_2 = a_2 + b_2\f$
 * 
 * @param[in] ctx         The context
 * @param[out] c1         Outcome of the first addition
 * @param[in]  a1         First operator of the first addition
 * @param[in]  b1         Second operator of the first addition
 * @param[out] c2         Outcome of the second addition
 * @param[in]  a2         First operator of the second addition
 * @param[in]  b2         Second operator of the second addition
 */
__host__ void poly_double_add(Context *ctx, poly_t *c1, poly_t *a1, poly_t *b1, poly_t *c2, poly_t *a2, poly_t *b2);

/**
 * @brief       Executes a polynomial subtraction.
 * 
 * Computes \f$c = a - b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
__host__ void poly_sub(Context *ctx, poly_t *c, poly_t *a, poly_t *b);

/**
 * @brief       Executes a polynomial multiplication.
 * 
 * Computes \f$c = a \times b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
__host__ void poly_mul(Context *ctx, poly_t *c, poly_t *a, poly_t *b);


/**
 * @brief       Executes a polynomial multiplication followed by a polynomial addition.
 * 
 * Computes \f$d = a \times b + c\f$.
 * 
 * @param ctx   The context
 * @param d     Outcome
 * @param a     First operand
 * @param b     Second operand
 * @param c     Third operand
 */
__host__ void poly_mul_add(Context *ctx, poly_t *d, poly_t *a, poly_t *b, poly_t *c);

__host__  void poly_dr2(
    Context *ctx,
    poly_t *ct21, // Outcome
    poly_t *ct22, // Outcome
    poly_t *ct23, // Outcome
    poly_t *ct01, // Operand 1
    poly_t *ct02, // Operand 1
    poly_t *ct11, // Operand 2
    poly_t *ct12);// Operand 2

/**
 * @brief Multiply each coefficient by and integer x
 * 
 * Computes \f$c = a \times x\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c     Outcome
 * @param[in]  a     The polynomial
 * @param[in]  x     The integer
 */
__host__  void poly_mul_int(Context *ctx, poly_t *c, poly_t *a, uint64_t x);
__host__  void poly_double_mul(
    Context *ctx, 
    poly_t *c,  poly_t *a,  uint64_t b,
    poly_t *f,  poly_t *d,  uint64_t e);

__host__  void poly_double_add_int(
    Context *ctx, 
    poly_t *b1,
    poly_t *a1,
    poly_t *b2,
    poly_t *a2,
    uint64_t x1,
    uint64_t x2);

/**
 * @brief Add a zero-degree polynomial
 * 
 * Computes \f$c = a \times x\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c     Outcome
 * @param[in]  a     The polynomial
 * @param[in]  x     The integer
 */
__host__  void poly_add_int(Context *ctx, poly_t *c, poly_t *a, uint64_t x);

/**
 * @brief Subtract a zero-degree polynomial
 * 
 * Computes \f$c = a \times x\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c     Outcome
 * @param[in]  a     The polynomial
 * @param[in]  x     The integer
 */
__host__  void poly_sub_int(Context *ctx, poly_t *c, poly_t *a, uint64_t x);

/**
 * @brief       Computes the basis extension from base Q to base QB inplace
 * 
 * @param[in]     ctx      the context
 * @param[out]    a_B      Outcome in base b
 * @param[in]     a_Q      Input in base q
 */
__host__ void poly_modup( Context *ctx, poly_t *a, poly_t *b, int level);

__host__ void poly_moddown( Context *ctx, poly_t *a, poly_t *b, int level);

/**                                                            *
 * @brief  The HPS method for computing \f$\rho\f$, used by CKKS's keygen
 *     
 * Section 4 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
 * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
 *
 * @param[in]     ctx      the context
 * @param[out]    c      Outcome of many polynomials in base q
 * @param[in]     a      Input in base q
 */
__host__ void poly_rho_ckks(Context *ctx, poly_t *c, poly_t *a);

__host__ void poly_ckks_rescale( Context *ctx,  poly_t *a, poly_t *b, int level = 1);


/**
 * @brief       Negate the coefficients of a polynomial a
 * 
 * @param ctx     The context
 * @param a    [description]
 */
__host__ void poly_negate(Context *ctx, poly_t *a);

/**
 * @brief       Right shift of each coefficient
 * 
 * @param ctx     The context
 * @param b       Output
 * @param a       Input
 * @param bits    Number of bits to shift
 */
__host__ void poly_right_shift(Context *ctx, poly_t *b, poly_t *a, int bits);

/**
 * @brief      Returns true if a == b, false otherwise.
 *
 * @param ctx     The context
 * @param      a     { parameter_description }
 * @param      b     { parameter_description }
 */
__host__ bool poly_are_equal(Context *ctx, poly_t *a, poly_t *b);

/**
 * @brief      Serializes a polynomial in a vector of ZZs
 *
 * @param      p     { parameter_description }
 *
 * @param ctx     The context
 * @return     { description_of_the_return_value }
 */
__host__ std::string poly_export(Context *ctx, poly_t *p);

/**
 * @brief      Returns a polynomial such that each coefficient vi lies in the ith-coefficient of v.
 *
 * @param ctx     The context
 * @param[in]  v     { parameter_description }
 */
// __host__ poly_t* poly_import_residues(Context *ctx, std::string v, int base = QBase);

/**
 * @brief      Serializes a polynomial in a vector of ZZs
 *
 * @param      p     { parameter_description }
 *
 * @param ctx     The context
 * @return     { description_of_the_return_value }
 */
// __host__ std::string poly_export_residues(Context *ctx, poly_t *p);

/**
 * @brief      Returns a polynomial such that each coefficient vi lies in the ith-coefficient of v.
 *
 * @param ctx     The context
 * @param[in]  v     { parameter_description }
 */
__host__ poly_t* poly_import(Context *ctx, std::string v);

/**
 * @brief      Compute the dot procut of a and b.
 * 
 * Computes \f$c = a \cdot b\f$.
 *
 * @param      ctx     { parameter_description }
 * @param      c     { parameter_description }
 * @param      a     { parameter_description }
 * @param      b     { parameter_description }
 * @param      k     { parameter_description }
 */
__host__ void poly_dot(
    Context *ctx, 
    poly_t *c,
    poly_t *a,
    poly_t *b,
    const int k);

/**
 * @brief      Return the infinity norm of p
 *
 * @param[in]  ctx     { parameter_description }
 * @param[in]  p     { parameter_description }
 */
// __host__ ZZ poly_infty_norm(Context *ctx, poly_t *p);

/**
 * @brief      Return the 2-norm of p
 *
 * @param[in]  ctx     { parameter_description }
 * @param[in]  p     { parameter_description }
 */
// __host__ RR poly_norm_2(Context *ctx, poly_t *p);

/**
 * @brief       Return the string representation of a residue
 * 
 * @param ctx     The context
 * @param a    the target poly_t
 * @param id    the id of the residue
 */
__host__ std::string poly_residue_to_string(Context *ctx, poly_t *a, int id = -1);

/**
 * @brief      Select a specific residue and overwrite everything else with it
 *
 * @param[in]  ctx     A context object
 * @param[in]  p     The operand
 * @param[in]  id     The residue
 */
// __host__ void poly_select_residue(Context *ctx, poly_t *p, int id);


/**
 * @brief      Returns an integer array with a particular residue of a.
 *
 * If b is allocated the outcome will be written on it, otherwise a new array will be allocated and the reference stored in b.
 *
 * @param      ctx   The context
 * @param[out]     b     The id-th residue of a
 * @param[int]      a     The polynomial
 * @param  id    The identifier
 *
 * @return     The id-th residue of a
 */
// __host__ uint64_t* poly_get_residue(Context *ctx, uint64_t* b, poly_t *a, int id);

/**
 * @brief      Returns the decomposition of a in its RNS base.
 *
 * @param      ctx   The context
 * @param[in]      a     The polynomial
 *
 * @return     The residues of a in a->base base.
 */
// __host__ uint64_t* poly_get_residues(Context *ctx, poly_t *a);

/**
 * @brief      Receives a polynomial represented in base QB and discard the q-residues.
 *
 * @param      ctx   The context
 * @param      p     { parameter_description }
 */
// __host__ void poly_discard_qbase(Context *ctx, poly_t *p);

#endif