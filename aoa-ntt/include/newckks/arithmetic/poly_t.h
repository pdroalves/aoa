#ifndef POLYNOMIAL_T
#define POLYNOMIAL_T

#include <map>
#include <vector>
#include <newckks/cuda/htrans/common.h>
#include <newckks/tool/context.h>

/**
 * @brief      Return the size in bytes required to store
 *             all the residues
 *             
 * @param base the base
 *
 * @return     the size in bytes required to store all the residues
 */
size_t poly_get_size(poly_bases b);

/**
 * @brief Initializes a poly_t object.
 * 
 * @param[in] ctx the context
 * @param[in, out] a The object
 * @param[in] base the base (the default value is QBase)
 */
void poly_init(Context *ctx, poly_t *a, poly_bases b = QBase);

/**
 * @brief Releases the memory related to a poly_t object
 * 
 * @param[in] a the object
 */
void poly_free(Context *ctx, poly_t *a);

/**
 * @brief Clear the memory related to a poly_t object without deallocating it.
 * 
 * Write 0's to all arrays.
 * 
 * @param ctx The context
 * @param a The object
 * 
 */
void poly_clear(Context *ctx, poly_t *a);

/**
 * @brief  Copy the coefficients from host's standard memory to device's global memory.
 * 
 * @param ctx   	The context
 * @param a     	the object
 * @param h_coefs   an array in the host memory with CUDAManager::nresidues_q residues
 */
void poly_copy_to_device(Context *ctx, poly_t *a, uint64_t *h_coefs);

/**
 * @brief  Copy the coefficients from device's global memory to host's main memory.
 * 
 * @param ctx   The context
 * @param a     the object
 */
uint64_t* poly_copy_to_host(Context *ctx, poly_t *a);

/**
 * @brief       Copy all coefficients from "a" to "b"
 * 
 * @param ctx   The context
 * @param b     Destiny
 * @param a     Source
 */
void poly_copy(Context *ctx, poly_t *b, poly_t *a);

bool poly_are_equal(Context *ctx, poly_t *a, poly_t *b);

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
void poly_add(Context *ctx, poly_t *c, poly_t *a, poly_t *b);

/**
 * @brief       Adds b to all coefficients
 * 
 * Computes \f$c = a + b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
void poly_add(Context *ctx, poly_t *c, poly_t *a, uint64_t b);

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
void poly_sub(Context *ctx, poly_t *c, poly_t *a, poly_t *b);

/**
 * @brief       Subtract b to all coefficients
 * 
 * Computes \f$c = a + b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
void poly_sub(Context *ctx, poly_t *c, poly_t *a, uint64_t b);

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
void poly_mul(Context *ctx, poly_t *c, poly_t *a, poly_t *b);

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
 void poly_mul(Context *ctx, poly_t *c, poly_t *a, uint64_t x);

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
void poly_add_add(Context *ctx, poly_t *c1, poly_t *a1, poly_t *b1, poly_t *c2, poly_t *a2, poly_t *b2);
void poly_mul_mul(Context *ctx, poly_t *c1, poly_t *a1, uint64_t b1, poly_t *c2, poly_t *a2, uint64_t b2);

 void poly_mul_add(Context *ctx, poly_t *d, poly_t *a, poly_t *b, poly_t *c);

 void poly_dr2(
    Context *ctx,
    poly_t *ct21, // Outcome
    poly_t *ct22, // Outcome
    poly_t *ct23, // Outcome
    poly_t *ct01, // Operand 1
    poly_t *ct02, // Operand 1
    poly_t *ct11, // Operand 2
    poly_t *ct12);// Operand 2

/**
 * @brief       Computes the basis extension from base Q to base QB inplace
 * 
 * @param[in]     ctx      the context
 * @param[out]    a_B      Outcome in base b
 * @param[in]     a_Q      Input in base q
 */
void poly_modup( Context *ctx, poly_t *a, poly_t *b, int level);

/**
 * @brief       Computes the basis extension from base QB to base Q inplace
 * 
 * @param[in]     ctx      the context
 * @param[out]    a_B      Outcome in base b
 * @param[in]     a_Q      Input in base q
 */
void poly_moddown( Context *ctx, poly_t *a, poly_t *b, int level);

void poly_rho( Context *ctx,  poly_t *b, poly_t *a);

void poly_rescale( Context *ctx,  poly_t *a, poly_t *b, int level = 1);

/**
 * @brief       Negate the coefficients of a polynomial a
 * 
 * @param ctx     The context
 * @param a    [description]
 */
void poly_negate(Context *ctx, poly_t *a);

/**
 * @brief      Serializes a polynomial in a vector of ZZs
 *
 * @param      p     { parameter_description }
 *
 * @param ctx     The context
 * @return     { description_of_the_return_value }
 */
std::string poly_export(Context *ctx, poly_t *p);

/**
 * @brief      Returns a polynomial such that each coefficient vi lies in the ith-coefficient of v.
 *
 * @param ctx     The context
 * @param[in]  v     { parameter_description }
 */
poly_t* poly_import(Context *ctx, std::string v);


/**
 * @brief       Return the string representation of a residue
 * 
 * @param ctx     The context
 * @param a    the target poly_t
 * @param id    the id of the residue
 */
std::string poly_residue_to_string(Context *ctx, poly_t *a, int id = -1);
#endif