#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include <newckks/arithmetic/poly_t.h>
#include <newckks/tool/context.h>
#include <newckks/cuda/manager.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
using namespace rapidjson;
typedef Document json;

/////////////////////////////
// Represents a ciphertext //
/////////////////////////////
struct ciphertext{
    poly_t c[2];
    int level;
    int slots;
    int64_t scale;
} typedef cipher_t;

void cipher_init(Context *ctx, cipher_t *c, int slots = 1);
void cipher_clear(Context *ctx, cipher_t *ct);
void cipher_free(Context *ctx, cipher_t *c);
void cipher_copy(Context *ctx, cipher_t *b, cipher_t *a);

std::string cipher_to_string(Context *ctx, cipher_t *c, int id = -1);

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