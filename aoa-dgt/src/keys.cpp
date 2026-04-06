#include <AOADGT/keys.h>

// Allocates memory for a Keys object
__host__ void keys_init(Context *ctx, Keys *k){
	poly_init(ctx, &k->pk->a);
	poly_init(ctx, &k->pk->b);
	poly_init(ctx, &k->evk->a, QBBase);
	poly_init(ctx, &k->evk->b, QBBase);
	poly_init(ctx, &k->cjk->a, QBBase);
	poly_init(ctx, &k->cjk->b, QBBase);
}
// Releases memory of a Keys object
__host__ void keys_free(Context *ctx, Keys *k){
	poly_free(ctx, &k->pk->a);
	poly_free(ctx, &k->pk->b);
	delete k->pk;

	poly_free(ctx, &k->evk->a);
	poly_free(ctx, &k->evk->b);
	delete k->evk;

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

	poly_free(ctx, &k->cjk->a);
	poly_free(ctx, &k->cjk->b);
	delete k->cjk;

}

SecretKey* ckks_new_sk(Context *ctx){
    SecretKey *sk = new SecretKey;
    poly_init(ctx, &sk->s, QBBase);

    ////////////////
    // Secret key //
    ////////////////
    // Low-norm secret key
    Sampler::sample(ctx, &sk->s, HAMMINGWEIGHT);
    sk->s.base = QBase;
    return sk;
};

PublicKey* ckks_new_pk(Context *ctx, SecretKey *sk){
	////////////////
	// Public key //
	////////////////
    PublicKey *pk = new PublicKey;
	poly_init(ctx, &pk->a);
	poly_init(ctx, &pk->b);

	poly_t e;
	poly_init(ctx, &e);
	Sampler::sample(ctx, &pk->a, UNIFORM);
	Sampler::sample(ctx, &pk->b, DISCRETE_GAUSSIAN);
	
	poly_bases aux = sk->s.base;
	sk->s.base = QBase; // This is a hack

	// b = [e - a*s]_q
	poly_mul(ctx, &e, &pk->a, &sk->s);
	poly_sub(ctx, &pk->b, &pk->b, &e);

	// Clean
	poly_free(ctx, &e);	

	sk->s.base = aux;

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
  // std::cout << "s1 0: " << poly_residue_to_string(ctx, s1, 0) << std::endl;
  // std::cout << "s1 1: " <<poly_residue_to_string(ctx, s1, 1) << std::endl;
  // std::cout << "s1 2: " <<poly_residue_to_string(ctx, s1, 2) << std::endl;
  // std::cout << "s2 0: " << poly_residue_to_string(ctx, s2, 0) << std::endl;
  // std::cout << "s2 1: " <<poly_residue_to_string(ctx, s2, 1) << std::endl;
  // std::cout << "s2 2: " <<poly_residue_to_string(ctx, s2, 2) << std::endl;
	poly_rho_ckks(ctx, &evk->b, s1);
  // std::cout << "sxsx 0: " << poly_residue_to_string(ctx, &evk->b, 0) << std::endl;
  // std::cout << "sxsx 1: " <<poly_residue_to_string(ctx, &evk->b, 1) << std::endl;
  // std::cout << "sxsx 2: " <<poly_residue_to_string(ctx, &evk->b, 2) << std::endl;

	// Sample
	Sampler::sample(ctx, &evk->a, UNIFORM);
	Sampler::sample(ctx, &e, DISCRETE_GAUSSIAN);
  // std::cout << "a'0: " << poly_residue_to_string(ctx, &evk->a, 0) << std::endl;;
  // std::cout << "a'1: " << poly_residue_to_string(ctx, &evk->a, 1) << std::endl;;
  // std::cout << "a'2: " << poly_residue_to_string(ctx, &evk->a, 2) << std::endl;
  // std::cout << "e0: " << poly_residue_to_string(ctx, &e, 0) << std::endl;
  // std::cout << "e1: " << poly_residue_to_string(ctx, &e, 1) << std::endl;
  // std::cout << "e2: " << poly_residue_to_string(ctx, &e, 2) << std::endl;
  // std::cout << "s20: " << poly_residue_to_string(ctx, s2, 0) << std::endl;
  // std::cout << "s21: " << poly_residue_to_string(ctx, s2, 1) << std::endl;
  // std::cout << "s22: " << poly_residue_to_string(ctx, s2, 2) << std::endl;

	// - a*s + e in base p and Rho_rns(s^2)  - a*s + e in base q
	poly_add(ctx, &evk->b, &evk->b, &e);
	poly_mul(ctx, &a, &evk->a, s2);
	poly_sub(ctx, &evk->b, &evk->b, &a);

	// Clean
	poly_free(ctx, &a);
	poly_free(ctx, &e);

	return evk;
}
