#ifndef CKKS_H
#define CKKS_H

#include <NTL/ZZ.h>
#include <AOADGT/ckkscontext.h>
#include <AOADGT/tool/encoder.h>
#include <AOADGT/arithmetic/ciphertext.h>
#include <AOADGT/arithmetic/polynomial.h>
#include <AOADGT/arithmetic/context.h>
#include <AOADGT/cuda/sampler.h>
#include <AOADGT/tool/log.h>
#include <omp.h>

///////////////////////
// CKKS cryptosystem //
//////////////////////

/**
 * @brief      Generate a set of keys
 *
 * @return     A struct with a tuple of keys
 */
Keys* ckks_keygen(CKKSContext *ctx, SecretKey *sk);

/**
 * @brief      CKKS encryption
 *
 * @param[out]      ct    The encryption of m
 * @param[in]       val     The input message
 *
 * @return     The encryption of m
 */
cipher_t* ckks_encrypt_poly(CKKSContext *ctx, cipher_t *ct, poly_t *p);

/**
 * @brief      CKKS encryption
 *
 * @param[in]      val     The input message
 *
 * @return     The encryption of m
 */
cipher_t* ckks_encrypt_poly(CKKSContext *ctx, poly_t *p);

/**
 * @brief      CKKS encryption
 *
 * @param[in]      val     The input message
 *
 * @return     The encryption of m
 */
cipher_t* ckks_encrypt(CKKSContext *ctx, complex<double>* val, int slots = 1, int empty_slots = 0);

/**
 * @brief      CKKS encryption
 *
 * @param[out]      ct    The encryption of m
 * @param[in]       val     The input message
 *
 * @return     The encryption of m
 */
cipher_t* ckks_encrypt(CKKSContext *ctx, cipher_t *ct, complex<double>* val, int slots = 1, int empty_slots = 0);

/**
 * @brief      CKKS encryption
 *
 * @param[in]      val     The input message
 *
 * @return     The encryption of m
 */
cipher_t* ckks_encrypt(CKKSContext *ctx, double val, int slots = 1, int empty_slots = 0);

/**
 * @brief      CKKS encryption
 *
 * @param[out]      ct    The encryption of m
 * @param[in]       val     The input message
 *
 * @return     The encryption of m
 */
cipher_t* ckks_encrypt(CKKSContext *ctx, cipher_t *ct, double val, int slots = 1, int empty_slots = 0);

/**
 * @brief      CKKS decryption
 *
 * @param[in]      c     The encryption of m
 *
 * @return     Message m
 */
 complex<double>* ckks_decrypt(CKKSContext *ctx, cipher_t *c, SecretKey *sk);

/**
 * @brief      CKKS decryption
 *
 * @param[in]      c     The encryption of m
 *
 * @return     Message m
 */
 complex<double>* ckks_decrypt(CKKSContext *ctx, complex<double>* val, cipher_t *c, SecretKey *sk);

/**
 * @brief      CKKS decryption
 *
 * @param[in]      c     The encryption of m
 *
 * @return     Message m
 */
poly_t* ckks_decrypt_poly(CKKSContext *ctx, poly_t *m, cipher_t *c, SecretKey *sk);


/**
 * @brief      CKKS rescaling
 *
 * @param      ctx   The context
 * @param      c     { parameter_description }
 */
void ckks_rescale(CKKSContext *ctx, cipher_t *c);

/**
 * @brief      CKKS's Homomorphic addition
 *
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 *
 * @return     Homomorphic addition of ct1 and ct2
 */
cipher_t* ckks_add(CKKSContext *ctx, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief      CKKS's Homomorphic addition
 *
 * @param[in]  ct3   Homomorphic addition of ct1 and ct2
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 */
void ckks_add(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);
/**
 * @brief      CKKS's Homomorphic subtraction
 *
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 *
 * @return     Homomorphic subtraction of ct1 and ct2
 */
cipher_t* ckks_sub(CKKSContext *ctx, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief      CKKS's Homomorphic subtraction
 *
 * @param[in]  ct3   Homomorphic subtraction of ct1 and ct2
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 */
void ckks_sub(CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);

/**
 * @brief      CKKS's Homomorphic multiplication
 *
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 * 
 * @return   \f$ct1 \times ct2\f$
 */

void ckks_mul( CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);

cipher_t* ckks_mul(CKKSContext *ctx, cipher_t *c1, cipher_t *c2);

void ckks_mul(CKKSContext *ctx, cipher_t *c2, cipher_t *c1, poly_t *val);

cipher_t* ckks_mul(CKKSContext *ctx, cipher_t *c1, poly_t *val);

void ckks_mul_without_rescale( CKKSContext *ctx, cipher_t *c2, cipher_t *c1, poly_t *val);

void ckks_mul_without_rescale( CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2);

cipher_t* ckks_mul_without_rescale(CKKSContext *ctx, cipher_t *c1, cipher_t *c2);

void ckks_add(CKKSContext *ctx, cipher_t *c2, cipher_t *c1, double val);

cipher_t* ckks_add(CKKSContext *ctx, cipher_t *c1, double val);

void ckks_mul(CKKSContext *ctx, cipher_t *c2, cipher_t *c1, double val);

cipher_t* ckks_mul(CKKSContext *ctx, cipher_t *c1, double val);

void ckks_mul_without_rescale(CKKSContext *ctx, cipher_t *c2, cipher_t *c1, double val);

cipher_t* ckks_mul_without_rescale(CKKSContext *ctx, cipher_t *c1, double val);

/**
 * @brief      CKKS's Homomorphic multiplication
 *
 * @param[in]  ct3   Homomorphic multiplication of ct1 and ct2
 * @param[in]  ct1   First operand
 * @param[in]  ct2   Second operand
 */
void ckks_mul(CKKSContext *ctx, cipher_t *ct3, cipher_t *ct1, cipher_t *ct2);

/**
 * @brief      Compute homomorphically c * c
 *
 * @param      ctx   The context
 * @param      c1    The c 1
 *
 * @return     { description_of_the_return_value }
 */
cipher_t* ckks_square(CKKSContext *ctx, cipher_t *c1);

/**
 * @brief      Compute homomorphically c * c
 *
 * @param      ctx   The context
 * @param      c1    The c 1
 *
 * @return     { description_of_the_return_value }
 */
void ckks_square(CKKSContext *ctx, cipher_t *c2, cipher_t *c1);

void ckks_inner_prod( CKKSContext *ctx, cipher_t *c3, cipher_t *c1, cipher_t *c2, const int size);
void ckks_inner_prod( CKKSContext *ctx, cipher_t *c3, cipher_t *c1, double   *c2, const int size);

template<typename T>
void ckks_batch_inner_prod(
    CKKSContext *ctx,
    cipher_t *c3,
    cipher_t *c1,
    T *c2);

void ckks_sumslots(CKKSContext *ctx, cipher_t *c2, cipher_t *c1);
cipher_t* ckks_sumslots(CKKSContext *ctx, cipher_t *c);

void ckks_power(CKKSContext *ctx, cipher_t *c2, cipher_t *c1, uint64_t x);

cipher_t* ckks_power(CKKSContext *ctx, cipher_t *c1, uint64_t x);

void ckks_power_of_2(CKKSContext *ctx, cipher_t *c2, cipher_t *c1, uint64_t x);

cipher_t* ckks_power_of_2(CKKSContext *ctx, cipher_t *c1, uint64_t x);

void ckks_eval_polynomial(CKKSContext *ctx, cipher_t *result, cipher_t *x, double *coeffs, int n);
cipher_t* ckks_eval_polynomial(CKKSContext *ctx, cipher_t *x, double *coeffs, int n);

void ckks_exp(CKKSContext *ctx, cipher_t *result, cipher_t *ct);
cipher_t* ckks_exp(CKKSContext *ctx, cipher_t *ct);
void ckks_log1minus(CKKSContext *ctx, cipher_t *result, cipher_t *ct);
cipher_t* ckks_log1minus(CKKSContext *ctx, cipher_t *ct);
void ckks_log(CKKSContext *ctx, cipher_t *result, cipher_t *ct, int steps = 10);
cipher_t* ckks_log(CKKSContext *ctx, cipher_t *ct, int steps = 10);

/**
 * Computes an approximation of sin(ct). The approximation works better when the plaintext is real.
 * @param ctx    [description]
 * @param result [description]
 * @param ct     [description]
 */
void ckks_sin(CKKSContext *ctx, cipher_t *result, cipher_t *ct);


/**
 * Computes an approximation of sin(ct). The approximation works better when the plaintext is real.
 * @param ctx    [description]
 * @param result [description]
 * @param ct     [description]
 */
cipher_t* ckks_sin(CKKSContext *ctx, cipher_t *ct);

/**
 * Computes an approximation of cos(ct). The approximation works better when the plaintext is real.
 * @param ctx    [description]
 * @param result [description]
 * @param ct     [description]
 */
void ckks_cos(CKKSContext *ctx, cipher_t *result, cipher_t *ct);


/**
 * Computes an approximation of cos(ct). The approximation works better when the plaintext is real.
 * @param ctx    [description]
 * @param result [description]
 * @param ct     [description]
 */
cipher_t* ckks_cos(CKKSContext *ctx, cipher_t *ct);
void ckks_sigmoid(CKKSContext *ctx, cipher_t *result, cipher_t *ct);
cipher_t* ckks_sigmoid(CKKSContext *ctx, cipher_t *ct);

/**
 * Computes 1/x for x \in (2,2)
 * 
 * @param ctx    [description]
 * @param result [description]
 * @param ct     [description]
 * @param steps  [description]
 */
void ckks_inverse(CKKSContext *ctx, cipher_t *result, cipher_t *ct, int steps = 10);
cipher_t* ckks_inverse(CKKSContext *ctx, cipher_t *ct, int steps = 10);
void ckks_rotate_right( CKKSContext *ctx, cipher_t *c1, cipher_t *c2, int rotSlots);
void ckks_rotate_left( CKKSContext *ctx, cipher_t *c1, cipher_t *c2, int rotSlots);
void ckks_conjugate( CKKSContext *ctx, cipher_t *c1, cipher_t *c2);
cipher_t* ckks_conjugate( CKKSContext * ctx, cipher_t *ct);
cipher_t* ckks_merge( CKKSContext * ctx, cipher_t *cts, int n);
void ckks_merge( CKKSContext * ctx, cipher_t *c_merged, cipher_t *cts, int n);

void ckks_discard_higher_slots( CKKSContext * ctx, cipher_t *ct);
void ckks_discard_slots_except( CKKSContext * ctx, cipher_t *ct, int idx);
#endif