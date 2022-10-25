#include <AOADGT/arithmetic/ciphertext.h>

__host__ void cipher_init(Context *ctx, cipher_t *c, int slots, poly_bases base){
	poly_init(ctx, &c->c[0], base);
	poly_init(ctx, &c->c[1], base);
	c->level = CUDAEngine::RNSPrimes.size() - 1;
	// c->scale = CUDAEngine::RNSPrimes[1];
	c->scale = ((uint64_t)1 << CUDAEngine::scalingfactor);
	c->slots = slots;
}

__host__ void cipher_clear(Context *ctx, cipher_t *c){
	if(c->c[0].state != NONINITIALIZED && c->c[1].state != NONINITIALIZED){
		poly_clear(ctx, &c->c[0]);
		poly_clear(ctx, &c->c[1]);
		c->level = CUDAEngine::RNSPrimes.size() - 1;
		c->scale = ((uint64_t)1 << CUDAEngine::scalingfactor);
	}else
		cipher_init(ctx, c);
}

__host__ void cipher_free(Context *ctx, cipher_t *c){
	poly_free(ctx, &c->c[0]);
	poly_free(ctx, &c->c[1]);
}

__host__ void cipher_copy(Context *ctx, cipher_t *b, cipher_t *a){
	poly_copy(ctx, &b->c[0], &a->c[0]);
	poly_copy(ctx, &b->c[1], &a->c[1]);
	b->level = a->level;
	b->slots = a->slots;
}

__host__ std::string cipher_to_string(Context *ctx, cipher_t *c){
	return std::string("c0: ") +
	poly_residue_to_string(ctx, &c->c[0]) +
	"\n" +
	"c1: " +
	poly_residue_to_string(ctx, &c->c[1]);
}

json cipher_export(Context *ctx, cipher_t *ct){
	std::string ct0 = poly_export(ctx, &ct->c[0]);
	std::string ct1 = poly_export(ctx, &ct->c[1]);

	json jcts;
	jcts.SetObject();
	jcts.AddMember("ct", Value{}.SetObject(), jcts.GetAllocator());
	jcts["ct"].AddMember("0",
		Value{}.SetString(ct0.c_str(),
		ct0.length(),
		jcts.GetAllocator()),
		jcts.GetAllocator());
	jcts["ct"].AddMember("1",
		Value{}.SetString(ct1.c_str(),
		ct1.length(),
		jcts.GetAllocator()),
		jcts.GetAllocator());

	json jprops;
	jprops.SetObject();
	jprops.AddMember("level", ct->level, jcts.GetAllocator());
	jprops.AddMember("scale", ct->scale, jcts.GetAllocator());
	jprops.AddMember("slots", ct->slots, jcts.GetAllocator());

	jcts.AddMember("props", jprops, jcts.GetAllocator());
	return jcts;
}

cipher_t* cipher_import(Context *ctx, const json & k){
	cipher_t *ct = new cipher_t;
	cipher_init(ctx, ct);

	poly_copy(ctx, &ct->c[0], poly_import(ctx, k["ct"]["0"].GetString()));
	poly_copy(ctx, &ct->c[1], poly_import(ctx, k["ct"]["1"].GetString()));

	ct->level = k["props"]["level"].GetInt();
	ct->scale = k["props"]["scale"].GetInt64();
	ct->slots = k["props"]["slots"].GetInt();

	return ct;
}