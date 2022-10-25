#include <newckks/ckks/ckkskeys.h>
#include <newckks/ckks/encoder.h>
#include <newckks/cuda/sampler.h>
#include <newckks/ckks/ckkscontext.h>

// Allocates memory for a CKKSKeychain object
__host__ void ckkskeychain_init(Context *ctx, CKKSKeychain *k){
	k->pk = new PublicKey();
	k->evk = new EvaluationKey();
	k->cjk = new ConjugationKey();

	poly_init(ctx, &k->pk->a);
	poly_init(ctx, &k->pk->b);
	poly_init(ctx, &k->evk->a, QBBase);
	poly_init(ctx, &k->evk->b, QBBase);
	poly_init(ctx, &k->cjk->a, QBBase);
	poly_init(ctx, &k->cjk->b, QBBase);
}

// Releases memory of a CKKSKeychain object
__host__ void ckkskeychain_free(Context *ctx, CKKSKeychain *k){
	if(k->pk){
		poly_free(ctx, &k->pk->a);
		poly_free(ctx, &k->pk->b);
		delete k->pk;
		k->pk = NULL;
	}

	if(k->evk){
		poly_free(ctx, &k->evk->a);
		poly_free(ctx, &k->evk->b);
		delete k->evk;
		k->evk = NULL;
	}

 	for (std::pair<int, RotationKey*> key : k->rtk_right){
		poly_free(ctx, &key.second->b);
		poly_free(ctx, &key.second->a);
 	}
	for (std::pair<int, RotationKey*> key : k->rtk_right)
		delete key.second;
	k->rtk_right.clear();

 	for (std::pair<int, RotationKey*> key : k->rtk_left){
		poly_free(ctx, &key.second->b);
		poly_free(ctx, &key.second->a);
 	}
	for (std::pair<int, RotationKey*> key : k->rtk_left)
		delete key.second;
	k->rtk_left.clear();

	if(k->cjk){
		poly_free(ctx, &k->cjk->a);
		poly_free(ctx, &k->cjk->b);
		delete k->cjk;
		k->cjk = NULL;
	}

}

SecretKey* ckks_new_sk(Context *ctx){
    SecretKey *sk = new SecretKey;
    poly_init(ctx, &sk->s, QBBase);

    ////////////////
    // Secret key //
    ////////////////
    // Low-norm secret key
    ctx->get_sampler()->sample_hw(&sk->s, QBBase);
    // sk->s.base = QBase;
    return sk;
}

__host__ void ckks_free_sk(Context *ctx, SecretKey *sk){
	poly_free(ctx, &sk->s);
}

PublicKey* ckks_new_pk(Context *ctx, SecretKey *sk){
	////////////////
	// Public key //
	////////////////
    PublicKey *pk = new PublicKey;
	poly_init(ctx, &pk->a, QBBase);
	poly_init(ctx, &pk->b, QBBase);

	poly_t e;
	poly_init(ctx, &e);
	ctx->get_sampler()->sample_uniform(&pk->a, QBBase);
	ctx->get_sampler()->sample_DG(&pk->b, QBBase);
	
	// b = [e - a*s]_q
	poly_mul(ctx, &e, &pk->a, &sk->s);
	poly_sub(ctx, &pk->b, &pk->b, &e);

	// Clean
	poly_free(ctx, &e);	

	return pk;
}

EvaluationKey* ckks_new_evk(Context *ctx, poly_t *s1, poly_t *s2){
	////////////////////
	// Evaluation key //
	////////////////////
	// Init in QBBase
	// 
    EvaluationKey *evk = new EvaluationKey;
	poly_init(ctx, &evk->a, QBBase);
	poly_init(ctx, &evk->b, QBBase);
	poly_t a,e;
	poly_init(ctx, &a, QBBase);
	poly_init(ctx, &e, QBBase);
	
	// // Compute rho //
	poly_rho(ctx, &evk->b, s1);

	// Sample
	ctx->get_sampler()->sample_uniform(&evk->a, QBBase);
	ctx->get_sampler()->sample_DG(&e, QBBase);

	// - a*s + e in base p and Rho_rns(s^2)  - a*s + e in base q
	poly_add(ctx, &evk->b, &evk->b, &e);
	poly_mul(ctx, &a, &evk->a, s2);
	poly_sub(ctx, &evk->b, &evk->b, &a);

	// Clean
	poly_free(ctx, &a);
	poly_free(ctx, &e);

	return evk;
}

EvaluationKey* ckks_new_mtk(Context *ctx, SecretKey *sk){

	poly_t s2;
	poly_init(ctx, &s2, QBBase);
	
	// This multiplication must happen in QBBase
	poly_mul(ctx, &s2, &sk->s, &sk->s);
	
	EvaluationKey *evk = ckks_new_evk(ctx, &s2, &sk->s);

	return evk;
}

std::map<int, RotationKey*> ckks_new_rtk_left(Context *ctx, SecretKey *sk){
	std::map<int, RotationKey*> rtk_left;

	poly_t s2;
	poly_init(ctx, &s2, QBBase);

	// Rotation key //
	for (int i = 1; i < CUDAManager::N; i *= 2){
		rotate_slots_left(ctx, &s2, &sk->s, i);
		rtk_left.insert(
			std::pair<int, RotationKey*>(i, ckks_new_evk(ctx, &s2, &sk->s))
			);
	}

	poly_free(ctx, &s2);

	return rtk_left;
}

std::map<int, RotationKey*> ckks_new_rtk_right(Context *ctx, SecretKey *sk){
	std::map<int, RotationKey*> rtk_right;

	poly_t s2;
	poly_init(ctx, &s2, QBBase);

	// Rotation key //
	for (int i = 1; i < CUDAManager::N; i *= 2){
		rotate_slots_right(ctx, &s2, &sk->s, i);
		rtk_right.insert(
			std::pair<int, RotationKey*>(i, ckks_new_evk(ctx, &s2, &sk->s))
			);
	}
	poly_free(ctx, &s2);

	return rtk_right;
}

ConjugationKey* ckks_new_cjk(Context *ctx, SecretKey *sk){
	poly_t s2;
	poly_init(ctx, &s2, QBBase);

	poly_copy(ctx, &s2, &sk->s);
	conjugate_slots(ctx, &s2);
	ConjugationKey *cjk = ckks_new_evk(ctx, &s2, &sk->s);

	return cjk;
}
