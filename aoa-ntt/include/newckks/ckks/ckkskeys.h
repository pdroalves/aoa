#ifndef KEYS_H
#define KEYS_H

#include <newckks/cuda/htrans/common.h>
#include <newckks/tool/context.h>
#include <newckks/arithmetic/poly_t.h>
#include <newckks/ckks/cipher_t.h>

// A composite of all types of keys
typedef struct{
    SecretKey *sk = NULL;
    PublicKey *pk = NULL;
    EvaluationKey *evk = NULL;
    std::map<int, RotationKey*> rtk_right;
    std::map<int, RotationKey*> rtk_left;
    ConjugationKey *cjk = NULL;
} CKKSKeychain;

void ckkskeychain_init(Context *ctx, CKKSKeychain *k);
void ckkskeychain_free(Context *ctx, CKKSKeychain *k);

SecretKey* ckks_new_sk(Context *ctx);
void ckks_free_sk(Context *ctx, SecretKey *sk);
PublicKey* ckks_new_pk(Context *ctx, SecretKey *sk);
EvaluationKey* ckks_new_evk(Context *ctx, poly_t *s1, poly_t *s2);

EvaluationKey* ckks_new_mtk(Context *ctx, SecretKey *sk);
std::map<int, RotationKey*> ckks_new_rtk_left(Context *ctx, SecretKey *sk);
std::map<int, RotationKey*> ckks_new_rtk_right(Context *ctx, SecretKey *sk);
ConjugationKey* ckks_new_cjk(Context *ctx, SecretKey *sk);

#endif
