#include <gtest/gtest.h>
#include <newckks/cuda/htrans/common.h>
#include <newckks/arithmetic/poly_t.h>
#include <newckks/cuda/manager.h>
#include <newckks/cuda/htrans/ntt.h>
#include <newckks/tool/version.h>
#include <newckks/cuda/sampler.h>
#include <newckks/ckks/ckks.h>
#include <newckks/ckks/ckkscontext.h>
#include <newckks/ckks/ckkskeys.h>
#include <newckks/ckks/encoder.h>
#include <newckks/coprimes.h>

#include <stdlib.h>
#include <cxxopts.hpp>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZX.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pEX.h>

using namespace NTL;

typedef struct{
	int k;
	int kl;
	int nphi;
	int prec;
} TestParams;

const unsigned NTESTS = 10;
const unsigned NTESTSINTENSE = 1000;
const float ERRBOUND = 0.005;

long double get_range(uint64_t scalingfactor){
	return (long double)(CUDAManager::RNSQPrimes[0] - 1) / ((uint64_t)1 << (scalingfactor + 1));
}

// Focus on the arithmetic
class TestPrimitives : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));

		// TestParams
		prec = (int)GetParam().prec;

	}

	__host__ void TearDown()
	{
		cudaDeviceReset();
	}
};

// Focus on the arithmetic
class TestNTTArithmetic : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	Context *ctx;

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		int nphi = (int)GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAManager::init(k, kl, nphi, prec, NTTTrans);// Init ClUDA
		ZZ q = CUDAManager::get_q_product();
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, CUDAManager::N, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);
	}

	__host__ void TearDown()
	{
		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();
		delete ctx;
		CUDAManager::destroy();
		cudaDeviceReset();
	}
};

class TestBasic : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	Context *ctx;

	public:
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		int nphi = (int)GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAManager::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAManager::get_q_product();
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, CUDAManager::N, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);
		
	}

	__host__ void TearDown()
	{
		delete ctx;
		CUDAManager::destroy();
		cudaDeviceReset();
	}
};

//
// Focus on Sampler
class TestSampler : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	Context *ctx;

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		int nphi = (int)GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAManager::init(k, kl, nphi, prec);// Init ClUDA
		ctx = new Context();
	}

	__host__ void TearDown()
	{
		delete ctx;
		CUDAManager::destroy();
		cudaDeviceReset();
	}
};

class TestCKKSBasics : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	CKKSContext *cipher;
	CKKSKeychain *keys;
	SecretKey *sk;

	public:
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		int nphi = (int)GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAManager::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAManager::get_q_product();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, CUDAManager::N, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// CKKS setup
        cipher = new CKKSContext();
  		sk = ckks_new_sk(cipher);

        keys = ckks_keygen(cipher, sk);
		
	}

	__host__ void TearDown()
	{
		ckks_free_sk(cipher, sk);
		delete keys;
		delete sk;
		delete cipher;
		CUDAManager::destroy();
		cudaDeviceReset();
	}
};

class TestCKKSAdv : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	CKKSContext *cipher;
	CKKSKeychain *keys;
	SecretKey *sk;

	public:
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		int nphi = (int)GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAManager::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAManager::get_q_product();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, CUDAManager::N, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// CKKS setup
        cipher = new CKKSContext();
  		sk = ckks_new_sk(cipher);

        keys = ckks_keygen(cipher, sk);
		
	}

	__host__ void TearDown()
	{
		ckks_free_sk(cipher, sk);
		delete keys;
		delete sk;
		delete cipher;
		CUDAManager::destroy();
		cudaDeviceReset();
	}
};

//////////////////////////
// Basic DGT arithmetic //
//////////////////////////

uint64_t rand64(uint64_t upperbound = (uint64_t)2e62) {
    // Assuming RAND_MAX is 2^32-1
    uint64_t r = rand();
    r = r<<32 | rand();
    return r % upperbound;
}

#define TEST_DESCRIPTION(desc) RecordProperty("description", desc)
TEST_P(TestPrimitives, mod)
{
	for(unsigned ntest = 0; ntest < 10 * NTESTS; ntest++){
		ZZ a = (to_ZZ(rand64()) << 64) + to_ZZ(rand64());
		a %= to_ZZ(2<<prec);

		uint128_t a_cupoly;
		a_cupoly.hi = conv<uint64_t>(a>>64);
		a_cupoly.lo = conv<uint64_t>(a);

		ZZ a_ntl = to_ZZ(a_cupoly.hi);
		a_ntl <<= 64;
		a_ntl += a_cupoly.lo;

		// Test for all the initialized coprimes
		for(int i = 0; i < CUDAManager::get_n_residues(QBBase); i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = mod(a_cupoly, i);
			ASSERT_EQ(x, a_ntl % to_ZZ(p));
		}

	}
}

TEST_P(TestPrimitives, addmod)
{
	for(unsigned ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(int i = 0; 
			i < CUDAManager::get_n_residues(QBBase); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(addmod(a_cupoly, b_cupoly, i), (a_ntl + b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestPrimitives, submod)
{
	for(unsigned ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(int i = 0; 
			i < CUDAManager::get_n_residues(QBBase); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(submod(a_cupoly, b_cupoly, i), (a_ntl - b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestPrimitives, mulmod)
{
	for(unsigned ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

	//	ASSERT_EQ(mulmod(a_cupoly, b_cupoly, 80), (a_ntl * b_ntl) % to_ZZ(COPRIMES_BUCKET[80]));

		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		a_ntl = to_ZZ(a_cupoly);
		b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(int i = 0; 
			i < CUDAManager::get_n_residues(QBBase); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(mulmod(a_cupoly, b_cupoly, i), (a_ntl * b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestPrimitives, powmod)
{
	for(unsigned ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(int i = 0; 
			i < CUDAManager::get_n_residues(QBBase); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = fast_pow(a_cupoly, b_cupoly, i);
			ASSERT_EQ(x, PowerMod(a_ntl, b_ntl, to_ZZ(p)));
		}

	}
}

/////////
// RNS //
/////////

// Export and import polynomials
// TEST_P(TestBasic, Serialize)
// {
// 	FAIL(); // TODO: There is something wrong with this
// 	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
// 		poly_t a;
// 		poly_init(ctx, &a);

// 		ctx->get_sampler()->sample_DG(&a);


// 		uint64_t *h_a = poly_copy_to_host(ctx, &a);

// 		// export
// 		std::string s = poly_export(ctx, &a);
// 		poly_t *b = poly_import(ctx, s);

// 		// assert
// 		ASSERT_TRUE(poly_are_equal(ctx, &a, b));

// 		poly_free(ctx, &a);
// 		poly_free(ctx, b);
// 		delete b;
// 	}
// }

//
// Tests polynomial addition
TEST_P(TestNTTArithmetic, Transform)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b;
		poly_init(ctx, &a, QBBase);
		poly_init(ctx, &b, QBBase);

		// Sample random polynomials
		ctx->get_sampler()->sample_uniform(&a, QBBase);

		// Compute b = INTT(NTT(a))
		COMMONEngine::execute(ctx, &a, FORWARD);
		poly_copy(ctx, &b, &a);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);

		for(int rid = 0; rid < CUDAManager::get_n_residues(QBBase); rid++)
			for(int i = 0; i < CUDAManager::N; i++){
				ZZ x = to_ZZ(h_a[i + rid * CUDAManager::N]);
				ZZ y = to_ZZ(h_b[i + rid * CUDAManager::N]);
				ASSERT_EQ(x, y)
				 << ntest << ") Fail at index " << i << " at rid " << rid;
			}

		free(h_a);
		free(h_b);
		poly_free(ctx, &a);
		poly_free(ctx, &b);
	}
}

// Tests polynomial addition
TEST_P(TestNTTArithmetic, Add)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		ctx->get_sampler()->sample_uniform(&a, QBase);
		ctx->get_sampler()->sample_uniform(&b, QBase);

		// Add
		poly_add(ctx, &c, &a, &b);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);

		for(int rid = 0; rid < CUDAManager::get_n_residues(a.base); rid++)
			for(int i = 0; i < CUDAManager::N; i++){
				ZZ x = to_ZZ(h_a[i + rid * CUDAManager::N]);
				ZZ y = to_ZZ(h_b[i + rid * CUDAManager::N]);
					ASSERT_EQ((x + y) % COPRIMES_BUCKET[rid], h_c[i + rid * CUDAManager::N])
					 << ntest << ") Fail at index " << i << " at rid " << rid;
			}

		free(h_a);
		free(h_b);
		free(h_c);
		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

//
// Tests polynomial subtraction
TEST_P(TestNTTArithmetic, Sub)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		ctx->get_sampler()->sample_uniform(&a, QBase);
		ctx->get_sampler()->sample_uniform(&b, QBase);

		// Add
		poly_sub(ctx, &c, &a, &b);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);

		for(int rid = 0; rid < CUDAManager::get_n_residues(a.base); rid++)
			for(int i = 0; i < CUDAManager::N; i++){
				int idx = i + rid * CUDAManager::N;

				ZZ x = to_ZZ(h_a[idx]);
				ZZ y = to_ZZ(h_b[idx]);
					ASSERT_EQ(
						(x - y) % COPRIMES_BUCKET[rid],
						h_c[idx] % COPRIMES_BUCKET[rid])
					 << ntest << ") Fail at index " << i << " at rid " << rid;
			}

		free(h_a);
		free(h_b);
		free(h_c);
		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

//
// Tests polynomial multiplication
TEST_P(TestNTTArithmetic, Mul)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b,c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		ctx->get_sampler()->sample_uniform(&a, QBase);
		ctx->get_sampler()->sample_uniform(&b, QBase);
		poly_mul(ctx, &c, &a, &b);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);

		for(int rid = 0; rid < CUDAManager::get_n_residues(a.base); rid++){
			ZZX ntl_a, ntl_b, ntl_c;
			for(int i = 0; i < CUDAManager::N; i++)
				SetCoeff(ntl_a, i, to_ZZ(h_a[i + rid * CUDAManager::N]));
			for(int i = 0; i < CUDAManager::N; i++)
				SetCoeff(ntl_b, i, to_ZZ(h_b[i + rid * CUDAManager::N]));

			ntl_c = ntl_a * ntl_b % conv<ZZX>(NTL_Phi);

			for(int i = 0; i < CUDAManager::N;i++)
				ASSERT_EQ(
					h_c[i + rid * CUDAManager::N],
					coeff(ntl_c, i) % COPRIMES_BUCKET[rid]
					) << ntest << ") Fail at index " << i << " at rid " << rid;
		}

		free(h_a);
		free(h_b);
		free(h_c);
		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

//
// Tests polynomial multiplication
TEST_P(TestNTTArithmetic, MulAdd)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c, d;
		poly_init(ctx, &a);
		poly_init(ctx, &b);
		poly_init(ctx, &c);
		poly_init(ctx, &d);

		// Sample random polynomials
		ctx->get_sampler()->sample_uniform(&a, QBase);
		ctx->get_sampler()->sample_uniform(&b, QBase);
		ctx->get_sampler()->sample_uniform(&c, QBase);
		
		poly_mul_add(ctx, &d, &a, &b, &c);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);
		uint64_t *h_d = poly_copy_to_host(ctx, &d);

		for(int rid = 0; rid < CUDAManager::get_n_residues(a.base); rid++){
			ZZX ntl_a, ntl_b, ntl_c, ntl_d;
			for(int i = 0; i < CUDAManager::N; i++)
				SetCoeff(ntl_a, i, to_ZZ(h_a[i + rid * CUDAManager::N]));
			for(int i = 0; i < CUDAManager::N; i++)
				SetCoeff(ntl_b, i, to_ZZ(h_b[i + rid * CUDAManager::N]));
			for(int i = 0; i < CUDAManager::N; i++)
				SetCoeff(ntl_c, i, to_ZZ(h_c[i + rid * CUDAManager::N]));

			ntl_d = (ntl_a * ntl_b + ntl_c) % conv<ZZX>(NTL_Phi);

			for(int i = 0; i < CUDAManager::N;i++)
				ASSERT_EQ(
					h_d[i + rid * CUDAManager::N],
					coeff(ntl_d, i) % COPRIMES_BUCKET[rid]
					) << ntest << ") Fail at index " << i << " at rid " << rid;
		}

		free(h_a);
		free(h_b);
		free(h_c);
		free(h_d);
		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
		poly_free(ctx, &d);
	}
}
// Tests polynomial multiplication by an 32 bits int
TEST_P(TestNTTArithmetic, MulInt)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, c;
		uint64_t b;
		poly_init(ctx, &a);
		poly_init(ctx, &c);

		// Sample random polynomials
		ctx->get_sampler()->sample_DG(&a, QBase);
		b = rand();

		poly_mul(ctx, &c, &a, b);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);
		
		for(int rid = 0; rid < CUDAManager::get_n_residues(a.base); rid++){
			ZZ_pX ntl_a, ntl_c;
			for(int i = 0; i < CUDAManager::N; i++)
				SetCoeff(ntl_a, i, conv<ZZ_p>(h_a[i + rid * CUDAManager::N]));

			ntl_c = ntl_a * b % conv<ZZ_pX>(NTL_Phi) ;

			for(int i = 0; i < CUDAManager::N;i++)
				ASSERT_EQ(
					h_c[i + rid * CUDAManager::N],
					conv<ZZ>(coeff(ntl_c, i)) % COPRIMES_BUCKET[rid]) << "Fail at index " << i;
		}

		free(h_a);
		free(h_c);
		poly_free(ctx, &a);
		poly_free(ctx, &c);
	}
}

//
// Tests samples from the uniform distribution
TEST_P(TestSampler, Uniform)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		ctx->get_sampler()->sample_uniform(&a, QBase);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);

		for(unsigned j = 0; j < CUDAManager::RNSQPrimes.size(); j++){
			ZZ sum = to_ZZ(0);
			for(int i = 0; i < CUDAManager::N; i++){
				ASSERT_LT(h_a[i + j * CUDAManager::N], COPRIMES_BUCKET[j]);
				sum += to_ZZ(h_a[i + j * CUDAManager::N]);
			}
			ASSERT_GT(sum, to_ZZ(0));
		}

		free(h_a);
		poly_free(ctx, &a);
	}
}

double compute_norm(uint64_t* a, int length, int64_t p){
	ZZ aux = to_ZZ(0);
	for(int i = 0; i < length; i++){
		ZZ v = to_ZZ(a[i]);

		v = (v < p/2 ? v : v - p);

		aux += v * v;
	}
	return conv<double>(sqrt(to_RR(aux)));
}

//
// Tests samples from the narrow distribution
TEST_P(TestSampler, Narrow)
{
	double avgnorm = 0;
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		ctx->get_sampler()->sample_narrow(&a, QBase);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);

		// Verify the narrow
		for(int i = 0; i < CUDAManager::N; i++){
			ASSERT_TRUE(
				h_a[i] == 0 ||
				h_a[i] == 1 ||
				h_a[i] == COPRIMES_BUCKET[0] - 1);
		}
		avgnorm += compute_norm(h_a, CUDAManager::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < CUDAManager::N; i++)
			for(int j = 0; j < CUDAManager::get_n_residues(a.base); j++){
				ASSERT_EQ(
					(int64_t) (h_a[i] <= 1? h_a[i] : -1),
					(int64_t) (h_a[i + j * CUDAManager::N] <= 1?
						h_a[i + j * CUDAManager::N] : -1)
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		free(h_a);
		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}

//
// Tests samples from the discrete gaussian distribution
TEST_P(TestSampler, DiscreteGaussian)
{
	double avgnorm = 0;
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		ctx->get_sampler()->sample_DG(&a, QBase);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		
		// Verify the range
		uint64_t acc = 0;
		for(int i = 0; i < CUDAManager::N; i++){
			ASSERT_LE(
				(int64_t) (h_a[i] < 100?
					h_a[i] : h_a[i] - COPRIMES_BUCKET[0]),
				GAUSSIAN_STD_DEVIATION * 10 + GAUSSIAN_BOUND);
			ASSERT_GE(
				(int64_t) (h_a[i] < 100?
					h_a[i] : h_a[i] - COPRIMES_BUCKET[0]),
				-(GAUSSIAN_STD_DEVIATION * 10 + GAUSSIAN_BOUND));
			acc += h_a[i];
		}
		ASSERT_GT(acc, 0);
		avgnorm += compute_norm(h_a, CUDAManager::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < CUDAManager::N; i++)
			for(int j = 0; j < CUDAManager::get_n_residues(a.base); j++){
				ASSERT_EQ(
					(int64_t) (h_a[i] < 100?
						h_a[i] :  h_a[i] - COPRIMES_BUCKET[0]),
					(int64_t) (h_a[i + j * CUDAManager::N] < 100?
						h_a[i + j * CUDAManager::N] :
						 h_a[i + j * CUDAManager::N] - COPRIMES_BUCKET[j])
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		free(h_a);
		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}

// Tests samples from the binary distribution
TEST_P(TestSampler, HammingWeight)
{
	double avgnorm = 0;
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		ctx->get_sampler()->sample_hw(&a, QBase);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);

		// Verify the hamming weight
		int count = 0;
		for(int i = 0; i < CUDAManager::N; i++){
			count += (h_a[i] != 0);
			ASSERT_TRUE(h_a[i] == 0 || h_a[i] == 1 || h_a[i] == COPRIMES_BUCKET[0] - 1);
		}
		ASSERT_EQ(count, std::min(HAMMINGWEIGHT, CUDAManager::N));
		avgnorm += compute_norm(h_a, CUDAManager::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < CUDAManager::N; i++)
			for(int j = 0; j < CUDAManager::get_n_residues(a.base); j++){
				ASSERT_EQ(
					(int64_t) (h_a[i] <= 1? h_a[i] : -1),
					(int64_t) (h_a[i + j * CUDAManager::N] <= 1?
						h_a[i + j * CUDAManager::N] : -1)
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		free(h_a);
		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}


TEST_P(TestCKKSBasics, EncodeDecode)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(prec), get_range(prec));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val = new std::complex<double>[slots];
	std::complex<double> *val_decoded = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){
		/////////////
		// Message //
		/////////////
		poly_t m;
		poly_init(cipher, &m);

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), distribution(generator)};

		/////////////
		// Encode //
		/////////////
		int64_t scale = -1;
		cipher->encode(&m, &scale, val, slots);
		cipher->decode(val_decoded, &m, scale, slots);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val[i]),
				real(val_decoded[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val[i]),
				imag(val_decoded[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			diff += abs(val[i] - val_decoded[i]);
		}
		EXPECT_LE(real(diff), ERRBOUND);
		EXPECT_LE(imag(diff), ERRBOUND);

		poly_free(cipher, &m);
	}
	diff /= slots * NTESTS;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val;
	delete[] val_decoded;
}


TEST_P(TestCKKSBasics, EncryptDecrypt)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(prec), get_range(prec));
	int slots = 1;
	// int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val = new std::complex<double>[slots];


	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), distribution(generator)};
		
		/////////////
		// Encrypt //
		/////////////
		cipher_t* ct = ckks_encrypt(cipher, val, slots);
		std::complex<double>* val_decoded = ckks_decrypt(cipher, ct, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val[i]),
				real(val_decoded[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val[i]),
				imag(val_decoded[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs(val[i] - val_decoded[i]);
		}
		EXPECT_LE(real(diff), ERRBOUND);
		EXPECT_LE(imag(diff), ERRBOUND);
		cipher_free(cipher, ct);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val;
}

TEST_P(TestCKKSBasics, Add)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(prec)/2, get_range(prec)/2);
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++){
			val1[i] = {distribution(generator), distribution(generator)};
			val2[i] = {distribution(generator), distribution(generator)};
		}

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);
		cipher_t *ct2 = ckks_encrypt(cipher, val2, slots);

		/////////
		// Add //
		/////////
		cipher_t *ct3 = ckks_add(cipher, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] + val2[i]),
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] + val2[i]),
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] + val2[i]) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct2;
		delete ct3;
		free(val3);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
	delete[] val2;
}

TEST_P(TestCKKSBasics, Sub)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(prec)/2, get_range(prec)/2);
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++){
			val1[i] = {distribution(generator), distribution(generator)};
			val2[i] = {distribution(generator), distribution(generator)};
		}

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);
		cipher_t *ct2 = ckks_encrypt(cipher, val2, slots);

		/////////
		// Add //
		/////////
		cipher_t *ct3 = ckks_sub(cipher, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] - val2[i]),
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] - val2[i]),
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] - val2[i]) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct2;
		delete ct3;
		free(val3);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
	delete[] val2;
}

TEST_P(TestCKKSBasics, Mul)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(prec)), sqrt(get_range(prec)));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++){
			val1[i] = {distribution(generator), distribution(generator)};
			val2[i] = {distribution(generator), distribution(generator)};
			// val1[i] = {0.1 * i, 0.2 * i + 1};
			// val2[i] = {0.1 * i, 0.2 * i + 1};
		}

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);
		cipher_t *ct2 = ckks_encrypt(cipher, val2, slots);

		/////////
		// Add //
		/////////
		cipher_t *ct3 = ckks_mul(cipher, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val2[i]),
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] * val2[i]),
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] * val2[i]) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct2;
		delete ct3;
		free(val3);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
	delete[] val2;
}

TEST_P(TestCKKSBasics, Square)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(prec)), sqrt(get_range(prec)));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val1[i] = {distribution(generator), distribution(generator)};

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		////////////
		// Square //
		///////////
		cipher_t *ct2 = ckks_square(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val2 = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val1[i]),
				real(val2[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] * val1[i]),
				imag(val2[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] * val1[i]) - (val2[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		delete ct1;
		delete ct2;
		free(val2);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
}

TEST_P(TestCKKSBasics, AddPlaintextDouble)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(prec)/2, get_range(prec)/2);
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];
	double val2;

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val1[i] = {distribution(generator), distribution(generator)};
		val2 = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		/////////
		// Add //
		/////////
		cipher_t *ct3 = ckks_add(cipher, ct1, val2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] + val2),
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] + val2),
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] + val2) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct3;
		free(val3);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
}

TEST_P(TestCKKSBasics, MulByConstantDouble)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(prec)/2), sqrt(get_range(prec)/2));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];
	double val2;

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val1[i] = {distribution(generator), distribution(generator)};
		val2 = distribution(generator)	;

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		/////////
		// Add //
		/////////
		cipher_t *ct3 = ckks_mul(cipher, ct1, val2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val2),
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] * val2),
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] * val2) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct3;
		free(val3);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
}



TEST_P(TestCKKSBasics, MulByConstantArray)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(prec)/2), sqrt(get_range(prec)/2));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++){
			val1[i] = {distribution(generator), distribution(generator)};
			val2[i] = {distribution(generator), distribution(generator)};
		}

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		/////////
		// Add //
		/////////
		cipher_t *ct3 = ckks_mul(cipher, ct1, val2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val2[i]),
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] * val2[i]),
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] * val2[i]) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct3;
		free(val3);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
}



TEST_P(TestCKKSBasics, MulByConstantPoly)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(prec)/2), sqrt(get_range(prec)/2));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++){
			val1[i] = {distribution(generator), distribution(generator)};
			val2[i] = {distribution(generator), distribution(generator)};
		}

        poly_t *p = new poly_t;
        poly_init(cipher, p);
        int64_t scale = -1;
        cipher->encode(p, &scale, val2, slots, 0);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		/////////
		// Mul //
		/////////
		cipher_t *ct3 = ckks_mul(cipher, ct1, p);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val2[i]),
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] * val2[i]),
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] * val2[i]) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct3;
		free(val3);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val1;
}

TEST_P(TestCKKSBasics, DivByConstant)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(prec)/2), sqrt(get_range(prec)/2));
	std::complex<double> diff = 0;
	int slots = CUDAManager::N>>1;
	std::complex<double> *val1 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTSINTENSE; N++){
		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val1[i] = distribution(generator);
		double val2 = distribution(generator);
		val2 += 0.1 * (abs(1.0 / val2) >= get_range(prec)/2);
		
		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		/////////
		// Mul //
		/////////
		cipher_t *ct2 = ckks_mul(cipher, ct1, 1.0 / val2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *val3 = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i]) / val2,
				real(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient (" << (1.0 / val2) << ")";
			ASSERT_NEAR(
				imag(val1[i]) / val2,
				imag(val3[i]),
				ERRBOUND)  << N << ") Fail for the " << i << "-th coefficient (" << (1.0 / val2) << ")";
			diff += abs((val1[i] / val2) - (val3[i]));
		}
		diff /= slots;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		delete ct1;
		delete ct2;
		free(val3);
	}
	diff /= NTESTS;
	std::cout << "Average error: " << diff << std::endl;
	free(val1);
}

TEST_P(TestCKKSBasics, RotateRight)
{      
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(prec), get_range(prec));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;

	//////////////
	// Sampling //
	//////////////
	std::complex<double> *val1 = new std::complex<double>[slots];
	for(int i = 0; i < slots; i++)
		val1[i] = {distribution(generator), distribution(generator)};

	/////////////
	// Encrypt //
	/////////////
	cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);
	cipher_t *ct2 = new cipher_t;
	cipher_init(cipher, ct2);

	for(int l = 1; l <= slots; l++){
		/////////
		// Rotate //
		/////////
		ckks_rotate_right(cipher, ct2, ct1, 1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i]),
				real(val3[(i+l) % slots]),
				0.01)  << "Fail to rotate by " << l << " for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i]),
				imag(val3[(i+l) % slots]),
				0.01)  << "Fail to rotate by " << l << " for the " << i << "-th coefficient";
		}
		free(val3);
    	std::swap(ct1, ct2);
	}

	cipher_free(cipher, ct1);
	cipher_free(cipher, ct2);
	free(val1);
	std::cout << "Average error: " << diff << std::endl;
}

TEST_P(TestCKKSBasics, RotateLeft)
{      
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(prec), get_range(prec));
	int slots = CUDAManager::N>>1;
	std::complex<double> diff = 0;

	//////////////
	// Sampling //
	//////////////
	std::complex<double> *val1 = new std::complex<double>[slots];
	for(int i = 0; i < slots; i++)
		val1[i] = {distribution(generator), distribution(generator)};

	/////////////
	// Encrypt //
	/////////////
	cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);
	cipher_t *ct2 = new cipher_t;
	cipher_init(cipher, ct2);

	for(int l = 1; l <= slots; l++){
		/////////
		// Rotate //
		/////////
		ckks_rotate_left(cipher, ct2, ct1, 1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double>* val3 = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i]),
				real(val3[(i < l? slots + (i - l) : i - l)]),
				0.01)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i]),
				imag(val3[(i < l? slots + (i - l) : i - l)]),
				0.01)  << "Fail for the " << i << "-th coefficient";
		}
		free(val3);
    	std::swap(ct1, ct2);
	}

	cipher_free(cipher, ct1);
	cipher_free(cipher, ct2);
	free(val1);
	std::cout << "Average error: " << diff << std::endl;
}

TEST_P(TestCKKSBasics, Conjugate)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), distribution(generator)};

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = new cipher_t;
		cipher_init(cipher, ct2);
		ckks_conjugate(cipher, ct2, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(real(conj(val[i])), real(result[i]), ERRBOUND)  << "Fail for the " << i << "-th slot at test " << N;
			ASSERT_NEAR(imag(conj(val[i])), imag(result[i]), ERRBOUND)  << "Fail for the " << i << "-th slot at test " << N;
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}
	delete[] val;
}

TEST_P(TestCKKSAdv, MulSequence)
{        
	int supported_depth = (CUDAManager::RNSQPrimes.size() - 1);
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-1, 1);
	int slots = CUDAManager::N>>1;

	std::complex<double> *expected = new std::complex<double>[slots];
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++){
			val1[i] = {1, 0};
			val2[i] = {1, 0};
			// val1[i] = {distribution(generator), distribution(generator)};
			// val2[i] = {distribution(generator), distribution(generator)};
		}

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);
		cipher_t *ct2 = ckks_encrypt(cipher, val2, slots);

		/////////
		// Mul //
		/////////
		std::copy_n(val2, slots, expected);
		for(int j = 0; j < supported_depth; j++){
			ckks_mul(cipher, ct2, ct2, ct1);

			ckks_decrypt(cipher, val2, ct2, sk);

			double diff = 0;
			for(int i = 0; i < slots; i++){
				expected[i] *= val1[i];

				ASSERT_NEAR(real(expected[i]), real(val2[i]), ERRBOUND) << " fail at index " << i << " on test " << N << " at depth " << j;
				ASSERT_NEAR(imag(expected[i]), imag(val2[i]), ERRBOUND) << " fail at index " << i << " on test " << N << " at depth " << j;
				diff += abs(expected[i] - val2[i]);
			}

		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		delete ct1;
		delete ct2;
	}
	delete[] val1;
	delete[] val2;
	delete[] expected;
}

TEST_P(TestCKKSAdv, InnerProductSingle)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(prec)/2), sqrt(get_range(prec)/2));
	int length = 4;

	for(int N = 0; N < 10; N++){

		//////////////
		// Sampling //
		//////////////
		std::complex<double> *val1 = new std::complex<double>[length];
		std::complex<double> *val2 = new std::complex<double>[length];
		std::complex<double> val;
		do{
			for(int i = 0; i < length; i++){
				val1[i] = {distribution(generator), distribution(generator)};
				val2[i] = {distribution(generator), distribution(generator)};
			}
			val = std::inner_product(
			&val1[0], &val1[length], &val2[0],
			(std::complex<double>){0.0, 0.0});
		} while(
			real(val) <= -get_range(prec) || real(val) >= get_range(prec) ||
			imag(val) <= -get_range(prec) || imag(val) >= get_range(prec));
		// std::cout << "Result expected: " << val << std::endl;
		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = new cipher_t[length];
		cipher_t *ct2 = new cipher_t[length];
		for(int i = 0; i < length; i++){
			cipher_init(cipher, &ct1[i]);
			cipher_init(cipher, &ct2[i]);

			ckks_encrypt(cipher, &ct1[i], &val1[i]);
			ckks_encrypt(cipher, &ct2[i], &val2[i]);
		}

		//////////////////
		// InnerProduct //
		//////////////////
		// CT
		cipher_t ct3;
		cipher->keys.sk = sk;
		ckks_inner_prod(cipher, &ct3, ct1, ct2, length);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *val3 = ckks_decrypt(cipher, &ct3, sk);

		///////////
		// Check //
		///////////
		ASSERT_NEAR(real(val), real(*val3), 10*ERRBOUND);
		ASSERT_NEAR(imag(val), imag(*val3), 10*ERRBOUND);

		for(int i = 0; i < length; i++){
			cipher_free(cipher, &ct1[i]);
			cipher_free(cipher, &ct2[i]);
		}
		cipher_free(cipher, &ct3);
		free(val1);
		free(val2);
	}
}

TEST_P(TestCKKSAdv, InnerProductBatch)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-1, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(int N = 0; N < 1; N++){

		//////////////
		// Sampling //
		//////////////
		std::complex<double> val;
		do{
			for(int i = 0; i < slots; i++){
				val1[i] = {distribution(generator), distribution(generator)};
				val2[i] = {distribution(generator), distribution(generator)};
			}
			val = std::inner_product(
			&val1[0],
			&val1[slots],
			&val2[0],
			(std::complex<double>){0.0, 0.0});
		} while(
			real(val) <= -get_range(prec) || real(val) >= get_range(prec) ||
			imag(val) <= -get_range(prec) || imag(val) >= get_range(prec));
		
		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);
		cipher_t *ct2 = ckks_encrypt(cipher, val2, slots);

		//////////////////
		// InnerProduct //
		//////////////////
		// CT
		cipher_t ct3;
		cipher_init(cipher, &ct3);
		cipher->keys.sk = sk;
		
		ckks_batch_inner_prod(cipher, &ct3, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *val3 = ckks_decrypt(cipher, &ct3, sk);

		///////////
		// Check //
		///////////
		double diff = 0;
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val),
				real(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val),
				imag(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			diff += abs(val - val3[i]);
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		cipher_free(cipher, &ct3);
		free(val3);
	}
	free(val1);
	free(val2);
}

TEST_P(TestCKKSAdv, InnerProductBatchPoly)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-1, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val1 = new std::complex<double>[slots];
	std::complex<double> *val2 = new std::complex<double>[slots];

	for(int N = 0; N < 1; N++){

		//////////////
		// Sampling //
		//////////////
		std::complex<double> val;
		do{
			for(int i = 0; i < slots; i++){
				val1[i] = {distribution(generator), distribution(generator)};
				val2[i] = {distribution(generator), distribution(generator)};
			}
			val = std::inner_product(
			&val1[0],
			&val1[slots],
			&val2[0],
			(std::complex<double>){0.0, 0.0});
		} while(
			real(val) <= -get_range(prec) || real(val) >= get_range(prec) ||
			imag(val) <= -get_range(prec) || imag(val) >= get_range(prec));
		
        poly_t *p = new poly_t;
        poly_init(cipher, p);
        int64_t scale = -1;
        cipher->encode(p, &scale, val2, slots, 0);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		//////////////////
		// InnerProduct //
		//////////////////
		// CT
		cipher_t ct3;
		cipher_init(cipher, &ct3);
		cipher->keys.sk = sk;
		
		ckks_batch_inner_prod(cipher, &ct3, ct1, p);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *val3 = ckks_decrypt(cipher, &ct3, sk);

		///////////
		// Check //
		///////////
		double diff = 0;
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val),
				real(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val),
				imag(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			diff += abs(val - val3[i]);
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, &ct3);
		free(val3);
	}
	free(val1);
	free(val2);
}

TEST_P(TestCKKSAdv, Sumslots)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	int powerbound = 5;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(
			-pow(get_range(prec), 1.0/(powerbound + 1)), pow(get_range(prec), 1.0/(powerbound + 1))
			);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		int x = (rand() % powerbound);
		std::complex<double> sum = 0;
		for(int i = 0; i < slots; i++){
			val[i] = {1.0, 1.0};
			// val[i] = {distribution(generator), distribution(generator)};
			sum += val[i];
			if(abs(sum) > get_range(prec)){
				sum -= val[i];
				val[i] = {0, 0};
				break;
			}
		}

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher->keys.sk = sk;
		cipher_t *ct2 = ckks_sumslots(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(real(sum), real(result[i]), ERRBOUND)  << "Fail for the " << i << "-th slots";;
			ASSERT_NEAR(imag(sum), imag(result[i]), ERRBOUND)  << "Fail for the " << i << "-th slots";;
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}
	delete[] val;
}

TEST_P(TestCKKSAdv, Power)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	int powerbound = 5;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(
			-pow(get_range(prec), 1.0/(powerbound + 1)), pow(get_range(prec), 1.0/(powerbound + 1))
			);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		int x = (rand() % powerbound);
		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), distribution(generator)};

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_power(cipher, ct1, x);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(real(pow(val[i], x)), real(result[i]), ERRBOUND);
			ASSERT_NEAR(imag(pow(val[i], x)), imag(result[i]), ERRBOUND);
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}
	delete[] val;
}

TEST_P(TestCKKSAdv, PowerOf2)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		int x = (1 << (rand() % 4));
		std::uniform_real_distribution<double> distribution = 
			std::uniform_real_distribution<double>(-pow(get_range(prec), 1.0/(x*2)), pow(get_range(prec), 1.0/(x*2)));

		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), distribution(generator)};

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_power(cipher, ct1, x);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(real(pow(val[i], x)), real(result[i]), ERRBOUND);
			ASSERT_NEAR(imag(pow(val[i], x)), imag(result[i]), ERRBOUND);
		}
		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}
	delete[] val;
}

TEST_P(TestCKKSAdv, evalpoly)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	int polydegree = 11;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-1, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), 0};
		double *coeffs = cipher->maclaurin_coeffs[SIN];

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_eval_polynomial(cipher, ct1, coeffs, polydegree);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int j = 0; j < slots; j++){
			std::complex<double> expected = 0;
			for(int i = polydegree-1; i >= 0; i--)
				expected = coeffs[i] + val[j] * expected;
			ASSERT_NEAR(real(expected), real(result[j]), ERRBOUND) << ": failed at slot "<< j << " for polydegree " << polydegree;
			ASSERT_NEAR(imag(expected), imag(result[j]), ERRBOUND) << ": failed at slot "<< j << " for polydegree " << polydegree;
		}
		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	delete[] val;
}

TEST_P(TestCKKSAdv, exp)
{

	if(prec <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = distribution(generator);

		// std::cout << "Will compute " << val << " ^ " << x << std::endl;
		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_exp(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		// std::cout << "exp(" << val << ") Got " << (*result) << " and expected " << exp(val) << std::endl;
		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
			ASSERT_NEAR(exp(real(val[i])), real(result[i]), ERRBOUND + ERRBOUND);

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}
	delete[] val;
}

TEST_P(TestCKKSAdv, sin)
{
	if(prec <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];
	
	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_sin(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
			ASSERT_NEAR(sin(real(val[i])), real(result[i]), ERRBOUND) << " Failure at slot " << i << " at test " << N;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	float x = 0;
	do{

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = x;

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_sin(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		if(
			abs(sin(real(val[0])) -  real(result[0])) > ERRBOUND ||
			abs(sin(real(val[slots/4])) -  real(result[slots/4])) > ERRBOUND ||
			abs(sin(real(val[2*slots/4])) -  real(result[2*slots/4])) > ERRBOUND ||
			abs(sin(real(val[3*slots/4])) -  real(result[3*slots/4])) > ERRBOUND
			)
			break;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);

		x += 0.1;
	}while(1);

	std::cout << "Could validate sin in the range [0," << x << ")" << std::endl;
}

TEST_P(TestCKKSAdv, cos)
{
	if(prec <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];
	
	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_cos(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		// std::cout << "Got " << (*result) << " and expected " << log(val) << std::endl;
		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
			ASSERT_NEAR(cos(real(val[i])), real(result[i]), ERRBOUND);

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	float x = 0;
	do{

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = x;

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_cos(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		if(
			abs(cos(real(val[0])) -  real(result[0])) > ERRBOUND ||
			abs(cos(real(val[slots/4])) -  real(result[slots/4])) > ERRBOUND ||
			abs(cos(real(val[2*slots/4])) -  real(result[2*slots/4])) > ERRBOUND ||
			abs(cos(real(val[3*slots/4])) -  real(result[3*slots/4])) > ERRBOUND
			)
			break;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);

		x += 0.1;
	}while(1);

	std::cout << "Could validate cos in the range [0," << x << ")" << std::endl;
}

TEST_P(TestCKKSAdv, sigmoid)
{
	if(prec <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];

	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_sigmoid(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
			ASSERT_NEAR((1 / (1 + exp(-real(val[i])))), real(result[i]), ERRBOUND);

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	float x = 0;
	do{

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = x;

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_sigmoid(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		if(abs(1 / (1 + exp(-real(val[0]))) -  real(result[0])) > ERRBOUND)
			break;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);

		x += 0.1;
	}while(1);

	std::cout << "Could validate sigmoid in the range [0," << x << ")" << std::endl;

}

TEST_P(TestCKKSAdv, log)
{
	if(prec <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0.1, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];
	
	for(unsigned N = 0; N < NTESTS; N++){
        cipher->keys.sk = sk;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_log(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(log(real(val[i])), real(result[i]), 0.5) << " Failure at slot " << i << " at test " << N;
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	float x = 0.1;
	do{

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = x;

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_log(cipher, ct1, CUDAManager::get_n_residues(QBase)-1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		if(abs(log(real(val[0])) -  real(result[0])) > 0.5)
				break;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);

		x += 0.1;
	}while(1);
	std::cout << "Could validate log in the range [0," << x << ")" << std::endl;
}

TEST_P(TestCKKSAdv, lnminus)
{
	if(prec <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];
	
	for(unsigned N = 0; N < NTESTS; N++){
        cipher->keys.sk = sk;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_log1minus(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
			ASSERT_NEAR(log(1-real(val[i])), real(result[i]), 0.5) << " Failure at slot " << i << " at test " << N << " log(1-"<< real(val[i])<<")";

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	float x = 0;
	do{

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = x;

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_log1minus(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		if(abs(log(1 - real(val[0])) -  real(result[0])) > ERRBOUND + ERRBOUND)
			break;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);

		x += 0.1;
	}while(1);
}

TEST_P(TestCKKSAdv, Inverse)
{
	if(prec <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0.1, 1);
	int slots = CUDAManager::N>>1;
	std::complex<double> *val = new std::complex<double>[slots];
	
	for(unsigned N = 0; N < NTESTS; N++){

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_inverse(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
				ASSERT_NEAR(1/real(val[i]), real(result[i]), 10);

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	float x = 0.05;
	do{

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = x;

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher_t *ct2 = ckks_inverse(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		if(abs(1.0 / (real(val[0]))  -  real(result[0])) > 0.1)
				break;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);

		x += 0.05;
	}while(1);
	std::cout << "Could validate the inverse in the range [0," << x << ")" << std::endl;
}

TEST_P(TestCKKSAdv, DiscardSlots)
{
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0.1, 10);
	int slots = (CUDAManager::N>>1);
	std::complex<double> *val = new std::complex<double>[slots];
	//////////////
	// Sampling //
	//////////////
	for(int i = 0; i < slots; i++)
		val[i] = distribution(generator);
	
	for(unsigned s_idx = 0; s_idx < slots; s_idx++){
		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct = ckks_encrypt(cipher, val, slots);
		cipher_t *aux = ckks_encrypt(cipher, (double)1, slots);

		////////////
		// Square //
		////////////
		//
		ckks_mul(cipher, ct, ct, aux);
		ckks_discard_slots_except(cipher, ct, s_idx);

		//////////////
		// Decrypt  //
		//////////////
		std::complex<double> *result = ckks_decrypt(cipher, ct, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
			ASSERT_NEAR(real(result[i]), (i == s_idx? real(val[i]) : 0), 1e-3) << "Failed for slot " << i << std::endl;

		cipher_free(cipher, ct);
		free(result);
	}
}

//
//Defines for which parameters set cuPoly will be tested.
//It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<TestParams> params = ::testing::Values(
	// {   k, kl, CUDAManager::N, prec},
	(TestParams){2, 2, 2048, 55},
	(TestParams){3, 3, 4096, 55},
	(TestParams){5, 5, 8192, 55},
	(TestParams){5, 5, 16384, 55}
	);

::testing::internal::ParamGenerator<TestParams> advparams = ::testing::Values(
	// {   k, kl, CUDAManager::N, prec},
	(TestParams){20, 20, 2048, 55},
	(TestParams){30, 30, 4096, 48},
	(TestParams){30, 30, 4096, 52},
	(TestParams){30, 30, 4096, 55},
	(TestParams){50, 50, 8192, 55},
	(TestParams){50, 50, 16384, 55}
	);

std::string printParamName(::testing::TestParamInfo<TestParams> p){
	TestParams params = p.param;

	return std::to_string(params.nphi) +
	"_k" + std::to_string(params.k) + "_kl" + std::to_string(params.kl) +
	"_prec" + std::to_string(params.prec);
}

INSTANTIATE_TEST_CASE_P(CKKSInstantiation,
	TestSampler,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(CKKSInstantiation,
	TestPrimitives,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(CKKSInstantiation,
	TestNTTArithmetic,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(CKKSInstantiation,
	TestBasic,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(CKKSInstantiation,
	TestCKKSBasics,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(CKKSInstantiation,
	TestCKKSAdv,
	advparams,
	printParamName
);

int main(int argc, char **argv) {

  //////////////////////////
  ////////// Google tests //
  //////////////////////////
  std::cout << "Testing version " << GET_VERSION() << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  
  return RUN_ALL_TESTS();
}
