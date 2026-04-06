#ifndef KEYS_H
#define KEYS_H
#include <AOADGT/arithmetic/polynomial.h>
#include <AOADGT/cuda/sampler.h>
#include <AOADGT/arithmetic/context.h>

/////////////////////
// Keys            //
/////////////////////

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
// 

// A composite of all types of keys
typedef struct{
    PublicKey *pk;
    EvaluationKey *evk;
    std::map<int, RotationKey*> rtk_right;
    std::map<int, RotationKey*> rtk_left;
    ConjugationKey *cjk;
} Keys;

void keys_init(Context *ctx, Keys *k);
void keys_free(Context *ctx, Keys *k);

SecretKey* ckks_new_sk(Context *ctx);
PublicKey* ckks_new_pk(Context *ctx, SecretKey *sk);
EvaluationKey* ckks_new_evk(Context *ctx, poly_t *s1, poly_t *s2);

#endif
