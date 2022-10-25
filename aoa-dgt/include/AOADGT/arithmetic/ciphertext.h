#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include <AOADGT/arithmetic/polynomial.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
using namespace rapidjson;
typedef Document json;

/////////////////////////////
// Represents a ciphertext //
/////////////////////////////
typedef struct{
    poly_t c[2];
    int level = 0;
    int slots = 1;
    int64_t scale = -1;
} cipher_t;

__host__ void cipher_init(Context *ctx, cipher_t *c, int slots = 1, poly_bases base = QBase);
__host__ void cipher_clear(Context *ctx, cipher_t *ct);
__host__ void cipher_free(Context *ctx, cipher_t *c);
__host__ void cipher_copy(Context *ctx, cipher_t *b, cipher_t *a);
__host__ void cipher_mult(
    poly_t *c3_star, 
    cipher_t *c1, 
    cipher_t *c2,
    const int nphi,
    const int t);
__host__ void DR2(
    Context *ctx,
    poly_t *c3_star, 
    cipher_t *c1, 
    cipher_t *c2);

__host__ std::string cipher_to_string(Context *ctx, cipher_t *c);

/**
 * @brief      Serializes a ciphertext
 *
 * @param      p     { parameter_description }
 *
 * @return     { description_of_the_return_value }
 */
json cipher_export(Context *ctx, cipher_t *ct);

/**
 * @brief      Imports a serialized ciphertext
 *
 * @param[in]  v     { parameter_description }
 *
 * @return     { description_of_the_return_value }
 */
cipher_t* cipher_import(Context *ctx, const json & k);

#endif