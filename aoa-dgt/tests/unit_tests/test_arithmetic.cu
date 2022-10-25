#include <gtest/gtest.h>
#include <AOADGT/settings.h>
#include <AOADGT/arithmetic/polynomial.h>
#include <AOADGT/cuda/sampler.h>
#include <AOADGT/cuda/cudaengine.h>
#include <AOADGT/cuda/dgt.h>
#include <stdlib.h>
#include <cxxopts.hpp>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZX.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pEX.h>

typedef struct{
	int k;
	int kl;
	int nphi;
	int prec;
} TestParams;

//int LOGLEVEL = INFO;
// int LOGLEVEL = DEBUG;
int LOGLEVEL = QUIET;
unsigned int NTESTS = 100;

// Focus on the arithmetic
class TestArithmetic : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAEngine::RNSProduct;
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// Samplers
		Sampler::init(ctx);
	}

	__host__ void TearDown()
	{
		delete ctx;
		Sampler::destroy();
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

class TestRNS : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAEngine::RNSProduct;
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// Samplers
		Sampler::init(ctx);
		
	}

	__host__ void TearDown()
	{
		delete ctx;
		Sampler::destroy();
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

class TestDGT : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only/

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAEngine::RNSProduct;
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// Samplers
		Sampler::init(ctx);
		
	}

	__host__ void TearDown()
	{
		delete ctx;
		Sampler::destroy();
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

//
// Focus on Sampler
class TestSampler : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ctx = new Context();

		Sampler::init(ctx);
	}

	__host__ void TearDown()
	{
		delete ctx;
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

//////////////////////////
// Basic DGT arithmetic //
//////////////////////////

uint64_t rand64(uint64_t upperbound = 18446744073709551615) {
    // Assuming RAND_MAX is 2^32-1
    uint64_t r = rand();
    r = r<<32 | rand();
    return r % upperbound;
}

#define TEST_DESCRIPTION(desc) RecordProperty("description", desc)
TEST_P(TestDGT, mod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){
		ZZ a = (to_ZZ(rand64()) << 64) + to_ZZ(rand64());
		a %= to_ZZ(2<<prec);

		uint128_t a_aoadgt;
		a_aoadgt.hi = conv<uint64_t>(a>>64);
		a_aoadgt.lo = conv<uint64_t>(a);

		ZZ a_ntl = to_ZZ(a_aoadgt.hi);
		a_ntl <<= 64;
		a_ntl += a_aoadgt.lo;

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = mod(a_aoadgt, i);
			ASSERT_EQ(x, a_ntl % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, addmod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_aoadgt, b_aoadgt;
		a_aoadgt = rand64() % (1 << prec);
		b_aoadgt = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_aoadgt);
		ZZ b_ntl = to_ZZ(b_aoadgt);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(addmod(a_aoadgt, b_aoadgt, i), (a_ntl + b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, submod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_aoadgt, b_aoadgt;
		a_aoadgt = rand64() % (1 << prec);
		b_aoadgt = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_aoadgt);
		ZZ b_ntl = to_ZZ(b_aoadgt);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(submod(a_aoadgt, b_aoadgt, i), (a_ntl - b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, mulmod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_aoadgt, b_aoadgt;
		a_aoadgt = 3246018949690369621;
		b_aoadgt = 31686739480305835;

		ZZ a_ntl = to_ZZ(a_aoadgt);
		ZZ b_ntl = to_ZZ(b_aoadgt);

	//	ASSERT_EQ(mulmod(a_aoadgt, b_aoadgt, 80), (a_ntl * b_ntl) % to_ZZ(COPRIMES_BUCKET[80]));

		a_aoadgt = rand64() % (1 << prec);
		b_aoadgt = rand64() % (1 << prec);

		a_ntl = to_ZZ(a_aoadgt);
		b_ntl = to_ZZ(b_aoadgt);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(mulmod(a_aoadgt, b_aoadgt, i), (a_ntl * b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, powmod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_aoadgt, b_aoadgt;
		a_aoadgt = rand64() % (1 << prec);
		b_aoadgt = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_aoadgt);
		ZZ b_ntl = to_ZZ(b_aoadgt);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = fast_pow(a_aoadgt, b_aoadgt, i);
			ASSERT_EQ(x, PowerMod(a_ntl, b_ntl, to_ZZ(p)));
		}

	}
}
////////////////////////////////////////////////////////////////////////////////
// GaussianIntegerArithmetic
////////////////////////////////////////////////////////////////////////////////


TEST_P(TestDGT, Transform)
{
	for(unsigned ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);

		// Compute b = INTT(NTT(a))
		DGTEngine::execute_dgt( ctx, &a, FORWARD);
		poly_copy(ctx, &b, &a);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);

		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++)
			for(int i = 0; i < CUDAEngine::N; i++){
				ZZ x = to_ZZ(h_a[i + rid * CUDAEngine::N]);
				ZZ y = to_ZZ(h_b[i + rid * CUDAEngine::N]);
				ASSERT_EQ(x, y)
				 << ntest << ") Fail at index " << i << " at rid " << rid;
			}

		free(h_a);
		free(h_b);
		poly_free(ctx, &a);
		poly_free(ctx, &b);
	}
}

TEST_P(TestDGT, GIAdd) {
	// Test for all the initialized coprimes
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++)
		for(unsigned int i = 0;
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size());
			i++){

		    GaussianInteger a = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger b = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger expected = {
		      conv<uint64_t>((to_ZZ(a.re) + to_ZZ(b.re)) % COPRIMES_BUCKET[i]),
		      conv<uint64_t>((to_ZZ(a.imag) + to_ZZ(b.imag)) % COPRIMES_BUCKET[i])
		    };
		    GaussianInteger received = GIAdd(a, b, i);

		    ASSERT_EQ(
		      received.re,
		      expected.re
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " + " << "(" << b.re << ", " << b.imag << ")";
		    ASSERT_EQ(
		      received.imag,
		      expected.imag
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " + " << "(" << b.re << ", " << b.imag << ")";
		}
}

TEST_P(TestDGT, GISub) {
	// Test for all the initialized coprimes
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){
		for(unsigned int i = 0;
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size());
			i++){

		    GaussianInteger a = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger b = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger expected = {
		      conv<uint64_t>((to_ZZ(a.re) - to_ZZ(b.re)) % COPRIMES_BUCKET[i]),
		      conv<uint64_t>((to_ZZ(a.imag) - to_ZZ(b.imag)) % COPRIMES_BUCKET[i])
		    };
		    GaussianInteger received = GISub(a, b, i);

		    ASSERT_EQ(
		      received.re,
		      expected.re
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " - " << "(" << b.re << ", " << b.imag << ")";
		    ASSERT_EQ(
		      received.imag,
		      expected.imag
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " - " << "(" << b.re << ", " << b.imag << ")";
		}
  	}
}

TEST_P(TestDGT, GIMul) {
	// Test for all the initialized coprimes
	for(unsigned int test = 0; test < 10 * NTESTS; test++)
		for(unsigned int i = 0;
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size());
			i++){

		    GaussianInteger a = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger b = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger expected = {
		      submod(
		        mulmod(a.re, b.re, i),
		        mulmod(a.imag, b.imag, i),
		        i
		        ),
		      addmod(
		        mulmod(a.imag, b.re, i),
		        mulmod(a.re, b.imag, i),
		        i
		        )
		    };
		    GaussianInteger received = GIMul(a, b, i);

		    ASSERT_EQ(
		      received.re,
		      expected.re
		      ) << test << ") Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re
				<< ", " << b.imag << ") for rid " << i << " ( " << COPRIMES_BUCKET[i] << ")";
		    ASSERT_EQ(
		      received.imag,
		      expected.imag
		      ) << test << ") Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re
				<< ", " << b.imag << ") for rid " << i << " ( " << COPRIMES_BUCKET[i] << ")";
		}
}

//
// Tests polynomial addition
TEST_P(TestArithmetic, Add)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);

		// Add
		poly_add(ctx, &c, &a, &b);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);

		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++)
			for(int i = 0; i < 2 * CUDAEngine::N; i++){
				int idx = i + rid * 2 * CUDAEngine::N;

				ZZ x = to_ZZ(h_a[idx]);
				ZZ y = to_ZZ(h_b[idx]);
				ASSERT_EQ((x + y) % COPRIMES_BUCKET[rid], h_c[idx])
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
TEST_P(TestArithmetic, Sub)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);

		// Sub
		poly_sub(ctx, &c, &a, &b);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);

		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++)
			for(int i = 0; i < 2 * CUDAEngine::N; i++){
				int idx = i + rid * 2 * CUDAEngine::N;

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
TEST_P(TestArithmetic, Mul)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b,c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);

		poly_mul(ctx, &c, &a, &b);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);

		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++){
			ZZX ntl_a, ntl_b, ntl_c;
			for(int i = 0; i < 2 * CUDAEngine::N; i++)
				SetCoeff(ntl_a, i, to_ZZ(h_a[i + rid * 2 * CUDAEngine::N]));
			for(int i = 0; i < 2 * CUDAEngine::N; i++)
				SetCoeff(ntl_b, i, to_ZZ(h_b[i + rid * 2 * CUDAEngine::N]));

			ntl_c = ntl_a * ntl_b % conv<ZZX>(NTL_Phi);

			for(int i = 0; i < 2 * CUDAEngine::N;i++)
				ASSERT_EQ(
					h_c[i + rid * 2 * CUDAEngine::N],
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

TEST_P(TestArithmetic, MulAdd)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c, d;
		poly_init(ctx, &a);
		poly_init(ctx, &b);
		poly_init(ctx, &c);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);
		Sampler::sample(ctx, &c, UNIFORM);

		poly_mul_add(ctx, &d, &a, &b, &c);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_b = poly_copy_to_host(ctx, &b);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);
		uint64_t *h_d = poly_copy_to_host(ctx, &d);

		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++){
			ZZX ntl_a, ntl_b, ntl_c, ntl_d;
			for(int i = 0; i < 2 * CUDAEngine::N; i++)
				SetCoeff(ntl_a, i, to_ZZ(h_a[i + rid * 2 * CUDAEngine::N]));
			for(int i = 0; i < 2 * CUDAEngine::N; i++)
				SetCoeff(ntl_b, i, to_ZZ(h_b[i + rid * 2 * CUDAEngine::N]));
			for(int i = 0; i < 2 * CUDAEngine::N; i++)
				SetCoeff(ntl_c, i, to_ZZ(h_c[i + rid * 2 * CUDAEngine::N]));

			ntl_d = (ntl_a * ntl_b + ntl_c) % conv<ZZX>(NTL_Phi);

			for(int i = 0; i < 2 * CUDAEngine::N;i++)
				ASSERT_EQ(
					h_d[i + rid * 2 * CUDAEngine::N],
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
TEST_P(TestArithmetic, MulByInt)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a,c;
		uint64_t b;
		poly_init(ctx, &a);

		// Sample random polynomials
		// int degree = (rand() % nphi);
		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);
		b = rand();
		
		poly_mul_int(ctx, &c, &a, b);
		
		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		uint64_t *h_c = poly_copy_to_host(ctx, &c);
		
		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++){
			ZZ_pX ntl_a, ntl_c;
			for(int i = 0; i < 2 * CUDAEngine::N; i++)
				SetCoeff(ntl_a, i, conv<ZZ_p>(h_a[i + rid * 2 * CUDAEngine::N]));

			ntl_c = ntl_a * b % conv<ZZ_pX>(NTL_Phi) ;

			for(int i = 0; i < 2 * CUDAEngine::N;i++)
				ASSERT_EQ(
					h_c[i + rid * 2 * CUDAEngine::N],
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
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);

		for(unsigned j = 0; j < CUDAEngine::RNSPrimes.size(); j++){
			ZZ sum = to_ZZ(0);
			for(int i = 0; i < 2 * CUDAEngine::N; i++){
				ASSERT_LT(h_a[i + j * 2 * CUDAEngine::N], COPRIMES_BUCKET[j]);
				sum += to_ZZ(h_a[i + j * 2 * CUDAEngine::N]);
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
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, NARROW);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);

		// Verify the narrow
		for(int i = 0; i < 2 * CUDAEngine::N; i++){
			ASSERT_TRUE(
				h_a[i] == 0 ||
				h_a[i] == 1 ||
				h_a[i] == COPRIMES_BUCKET[0] - 1);
		}
		avgnorm += compute_norm(h_a, 2 * CUDAEngine::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < 2 * CUDAEngine::N; i++)
			for(int j = 0; j < CUDAEngine::get_n_residues(QBase); j++){
				ASSERT_EQ(
					(int64_t) (h_a[i] <= 1? h_a[i] : -1),
					(int64_t) (h_a[i + j * 2 * CUDAEngine::N] <= 1?
						h_a[i + j * 2 * CUDAEngine::N] : -1)
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
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);
		
		// Verify the range
		uint64_t acc = 0;
		for(int i = 0; i < 2 * CUDAEngine::N; i++){
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
		avgnorm += compute_norm(h_a, 2 * CUDAEngine::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < 2 * CUDAEngine::N; i++)
			for(int j = 0; j < CUDAEngine::get_n_residues(QBase); j++){
				ASSERT_EQ(
					(int64_t) (h_a[i] < 100?
						h_a[i] :  h_a[i] - COPRIMES_BUCKET[0]),
					(int64_t) (h_a[i + j * 2 * CUDAEngine::N] < 100?
						h_a[i + j * 2 * CUDAEngine::N] :
						 h_a[i + j * 2 * CUDAEngine::N] - COPRIMES_BUCKET[j])
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
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, HAMMINGWEIGHT);

		uint64_t *h_a = poly_copy_to_host(ctx, &a);

		// Verify the hamming weight
		int count = 0;
		for(int i = 0; i < 2 * CUDAEngine::N; i++){
			count += (h_a[i] != 0);
			ASSERT_TRUE(h_a[i] == 0 || h_a[i] == 1 || h_a[i] == COPRIMES_BUCKET[0] - 1);
		}
		ASSERT_EQ(count, std::min((int)HAMMINGWEIGHT, 2 * CUDAEngine::N));
		avgnorm += compute_norm(h_a, 2 * CUDAEngine::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < 2 * CUDAEngine::N; i++)
			for(int j = 0; j < CUDAEngine::get_n_residues(QBase); j++){
				ASSERT_EQ(
					(int64_t) (h_a[i] <= 1? h_a[i] : -1),
					(int64_t) (h_a[i + j * 2 * CUDAEngine::N] <= 1?
						h_a[i + j * 2 * CUDAEngine::N] : -1)
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		free(h_a);
		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}

//
//Defines for which parameters set AOADGT will be tested.
//It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<TestParams> params = ::testing::Values(
	// {   k, kl, nphi, prec},
	// (TestParams){2, 5, 32, 40},
	// (TestParams){2, 5, 128, 40},
	// (TestParams){2, 5, 2048, 40},
	// (TestParams){3, 5, 4096, 40},
	// (TestParams){5, 5, 8192, 40},
	// (TestParams){5, 5, 16384, 40},
	//(TestParams){40, 41, 32, 55},
	//(TestParams){2, 5, 32, 45},
	(TestParams){2, 5, 128, 45},
	(TestParams){2, 5, 2048, 45},
	(TestParams){3, 5, 4096, 45},
	(TestParams){5, 5, 8192, 45},
	(TestParams){5, 5, 16384, 45},
	(TestParams){2, 5, 32, 48},
	(TestParams){2, 5, 128, 48},
	(TestParams){2, 5, 2048, 48},
	(TestParams){3, 5, 4096, 48},
	(TestParams){5, 5, 8192, 48},
	(TestParams){5, 5, 16384, 48},
	// (TestParams){2, 5, 32, 52},
	// (TestParams){2, 5, 128, 52},
	// (TestParams){2, 5, 2048, 52},
	// (TestParams){3, 5, 4096, 52},
	// (TestParams){5, 5, 8192, 52},
	// (TestParams){5, 5, 16384, 52},
	//(TestParams){2, 5, 32, 55},
	(TestParams){2, 5, 128, 55},
	(TestParams){2, 5, 2048, 55},
	(TestParams){3, 5, 4096, 55},
	(TestParams){5, 5, 8192, 55},
	(TestParams){5, 5, 16384, 55}
	);

std::string printParamName(::testing::TestParamInfo<TestParams> p){
	TestParams params = p.param;

	return std::to_string(params.nphi) +
	"_k" + std::to_string(params.k) + "_kl" + std::to_string(params.kl) +
	"_prec" + std::to_string(params.prec);
}

INSTANTIATE_TEST_CASE_P(AOADGTCKKSInstantiation,
	TestArithmetic,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(AOADGTCKKSInstantiation,
	TestRNS,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(AOADGTCKKSInstantiation,
	TestDGT,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(AOADGTCKKSInstantiation,
	TestSampler,
	params,
	printParamName
);

int main(int argc, char **argv) {

  //////////////////////////
  ////////// Google tests //
  //////////////////////////
  std::cout << "Testing AOADGT " << GET_AOADGT_VERSION() << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  
  return RUN_ALL_TESTS();
}
