#include <stdlib.h>
#include <gtest/gtest.h>
#include <AOADGT/settings.h>
#include <AOADGT/arithmetic/polynomial.h>
#include <AOADGT/tool/version.h>
#include <AOADGT/tool/version.h>
#include <AOADGT/ckkscontext.h>
#include <AOADGT/ckks.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pEX.h>
#include <rapidjson/prettywriter.h>

typedef struct{
	int nphi;
	int k;
	int kl;
	int scalingfactor;
} TestParams;

//const int LOGLEVEL = INFO;
const int LOGLEVEL = QUIET;
const int NTESTS = 100;
const int NTESTSINTENSE = 1000;
const float ERRBOUND = 0.005;

long double get_range(int scalingfactor){
	return (long double)(CUDAEngine::RNSPrimes[0] - 1) / ((uint64_t)1 << (scalingfactor + 1));
}

class TestCKKSBasics : public ::testing::TestWithParam<TestParams> {
	protected:
    CKKSContext* m_cipher;
	ZZ_pX NTL_Phi;
	vector<Keys*> m_keys;
	CUDAParams p;
	SecretKey *sk;

	public:
	__host__ void SetUp(){
		srand(0);
		NTL::SetSeed(to_ZZ(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		p.k = (int)GetParam().k;
		p.kl = (int)GetParam().kl;
		p.nphi = (int)GetParam().nphi;
		p.pt = (int)GetParam().scalingfactor;

		// Init
        CUDAEngine::init(p);// Init CUDA

		ZZ_p::init(CUDAEngine::RNSProduct);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, p.nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);


		// CKKS setup
        m_cipher = new CKKSContext();
        Sampler::init(m_cipher);
  		sk = ckks_new_sk(m_cipher);
        m_keys.push_back(ckks_keygen(m_cipher, sk));
        m_cipher->sk = sk;
	}

	__host__ void TearDown(){
		cudaDeviceSynchronize();
		cudaCheckError();

		Sampler::destroy();
		CUDAEngine::destroy();

		cudaDeviceReset();
		cudaCheckError();
	}
};

class TestCKKSBasicHomomorphism : public ::testing::TestWithParam<TestParams> {
	protected:
    CKKSContext* m_cipher;
	ZZ_pX NTL_Phi;
	Keys* keys;
	CUDAParams p;
	SecretKey *sk;

	public:
	__host__ void SetUp(){
		srand(0);
		NTL::SetSeed(to_ZZ(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		p.k = (int)GetParam().k;
		p.kl = (int)GetParam().kl;
		p.nphi = (int)GetParam().nphi;
		p.pt = (int)GetParam().scalingfactor;

		// Init
        CUDAEngine::init(p);// Init CUDA

		ZZ_p::init(CUDAEngine::RNSProduct);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, p.nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);
		
		// CKKS setup
        m_cipher = new CKKSContext();
        Sampler::init(m_cipher);
  		sk = ckks_new_sk(m_cipher);
        keys = ckks_keygen(m_cipher, sk);
        m_cipher->sk = sk;
	}

	__host__ void TearDown(){
		poly_free(m_cipher, &sk->s);
		keys_free(m_cipher, keys);

		cudaDeviceSynchronize();
		cudaCheckError();

		delete sk;
		delete keys;
 		delete m_cipher;

		Sampler::destroy();
		CUDAEngine::destroy();

		cudaDeviceReset();
		cudaCheckError();
	}
};

class TestCKKSAdvHomomorphism : public ::testing::TestWithParam<TestParams> {
	protected:
    CKKSContext* m_cipher;
	ZZ_pX NTL_Phi;
	Keys* keys;
	CUDAParams p;
	SecretKey *sk;

	public:
	__host__ void SetUp(){
		srand(0);
		NTL::SetSeed(to_ZZ(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		p.k = (int)GetParam().k;
		p.kl = (int)GetParam().kl;
		p.nphi = (int)GetParam().nphi;
		p.pt = (int)GetParam().scalingfactor;

		// Init
        CUDAEngine::init(p);// Init CUDA

		ZZ_p::init(CUDAEngine::RNSProduct);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, p.nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// CKKS setup
        m_cipher = new CKKSContext();
        Sampler::init(m_cipher);
  		sk = ckks_new_sk(m_cipher);
        keys = ckks_keygen(m_cipher, sk);
        m_cipher->sk = sk;

        // Scaling factor
    	Logger::getInstance()->log_info(
        	("Supported range: (-" +
        	std::to_string(get_range(p.pt)) + ", " + std::to_string(get_range(p.pt)) +
        	")").c_str()
        	);
	}

	__host__ void TearDown(){
		poly_free(m_cipher, &sk->s);
		keys_free(m_cipher, keys);

		cudaDeviceSynchronize();
		cudaCheckError();

		delete sk;
		delete keys;
 		delete m_cipher;

		Sampler::destroy();
		CUDAEngine::destroy();

		cudaDeviceReset();
		cudaCheckError();
	}
};

TEST_P(TestCKKSBasics, EncodeDecodeSingle)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt), get_range(p.pt));

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;
		/////////////
		// Message //
		/////////////
		poly_t m;
		poly_init(cipher, &m);

		//////////////
		// Sampling //
		//////////////
		complex<double> val = {distribution(generator), distribution(generator)};
 		complex<double> val_decoded;		

		/////////////
		// Encode //
		/////////////
		int64_t scale = -1;
		cipher->encodeSingle(&m, &scale, val);
		cipher->decodeSingle(&val_decoded, &m, scale);

		///////////
		// Check //
		///////////
		ASSERT_NEAR(real(val), real(val_decoded), ERRBOUND);
		ASSERT_NEAR(imag(val), imag(val_decoded), ERRBOUND);

		poly_free(cipher, &m);
	}
}

TEST_P(TestCKKSBasics, EncodeDecodeBatch)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt), get_range(p.pt));
	int slots = p.nphi/2;
	complex<double> diff = 0;
	complex<double> *val = new complex<double>[slots];
	complex<double> *val_decoded = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;
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
		cipher->encode(&m, &scale, val, slots, 0);
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
		std::uniform_real_distribution<double>(-get_range(p.pt), get_range(p.pt));
	int slots = p.nphi/2;
	complex<double> diff = 0;
	complex<double> *val = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++){
			val[i] = {distribution(generator), distribution(generator)};
		}
		
		/////////////
		// Encrypt //
		/////////////
		cipher_t* ct = ckks_encrypt(cipher, val, slots);
		complex<double>* val_decoded = ckks_decrypt(cipher, ct, sk);

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

		cipher_free(cipher, ct);
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	delete[] val;
}

TEST_P(TestCKKSBasicHomomorphism, Add)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt)/2, get_range(p.pt)/2);
	int slots = p.nphi/2;
	complex<double> diff = 0;
	complex<double> *val1 = new complex<double>[slots];
	complex<double> *val2 = new complex<double>[slots];

	for(int N = 0; N < NTESTSINTENSE; N++){
        CKKSContext *cipher = m_cipher;

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
		complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] + val2[i]),
				real(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] + val2[i]),
				imag(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
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


TEST_P(TestCKKSBasicHomomorphism, Sub)
{        

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt)/2, get_range(p.pt)/2);
	int slots = p.nphi/2;
	complex<double> diff = 0;
	complex<double> *val1 = new complex<double>[slots];
	complex<double> *val2 = new complex<double>[slots];

	for(int N = 0; N < NTESTSINTENSE; N++){
        CKKSContext *cipher = m_cipher;

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
		// Sub //
		/////////
		cipher_t *ct3 = ckks_sub(cipher, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] - val2[i]),
				real(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] - val2[i]),
				imag(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
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
	free(val1);
	free(val2);
}

TEST_P(TestCKKSBasicHomomorphism, Mul)
{        
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = p.nphi/2;
	complex<double> diff = 0;
	complex<double> *val1 = new complex<double>[slots];
	complex<double> *val2 = new complex<double>[slots];

	for(int N = 0; N < NTESTSINTENSE; N++){
		// std::cout << "Test " << N << std::endl;
        CKKSContext *cipher = m_cipher;

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
		cipher_t *ct3 = ckks_mul(cipher, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		complex<double>* val3 = ckks_decrypt(cipher, ct3, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val2[i]),
				real(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th slot at test " << N;
			ASSERT_NEAR(
				imag(val1[i] * val2[i]),
				imag(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th slot at test " << N;
			diff += abs((val1[i] * val2[i]) - (val3[i]));
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		cipher_free(cipher, ct3);
		delete ct1;
		delete ct2;
		delete ct3;
		delete[] val3;
	}
	diff /= NTESTS * slots;
	std::cout << "Average error: " << diff << std::endl;
	free(val1);
	free(val2);
}

TEST_P(TestCKKSBasicHomomorphism, Square)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(p.pt)) / 2, sqrt(get_range(p.pt)) / 2);
	int slots = CUDAEngine::N;
	complex<double> *val1 = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

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
		////////////
		cipher_t *ct2 = ckks_square(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *val2 = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		// std::cout << val1 << " ^2 == " << (val1 * val1) << "==" << (val1 * val1) << " =? " << (*val2) << std::endl;
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val1[i]),
				real(val2[i]),
				ERRBOUND + ERRBOUND) << "Fail for the " << i << "-th slot at test " << N;
			ASSERT_NEAR(
				imag(val1[i] * val1[i]),
				imag(val2[i]),
				ERRBOUND + ERRBOUND) << "Fail for the " << i << "-th slot at test " << N;
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(val2);
	}
	free(val1);	
}

TEST_P(TestCKKSBasicHomomorphism, AddByConstant)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt)/2, get_range(p.pt)/2);
	complex<double> diff = 0;
	int slots = CUDAEngine::N;
	complex<double> *val1 = new complex<double>[slots];

	for(int N = 0; N < NTESTSINTENSE; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val1[i] = {distribution(generator), distribution(generator)};
		double val2 = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		/////////
		// Mul //
		/////////
		cipher_t *ct2 = ckks_add(cipher, ct1, val2);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *val3 = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] + val2),
				real(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] + val2),
				imag(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] + val2) - (val3[i]));
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


TEST_P(TestCKKSBasicHomomorphism, MulByConstant)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(p.pt)/2), sqrt(get_range(p.pt)/2));
	complex<double> diff = 0;
	int slots = CUDAEngine::N;
	complex<double> *val1 = new complex<double>[slots];

	for(int N = 0; N < NTESTSINTENSE; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val1[i] = {distribution(generator), distribution(generator)};
		double val2 = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val1, slots);

		/////////
		// Mul //
		/////////
		cipher_t *ct2 = ckks_mul(cipher, ct1, val2);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *val3 = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			ASSERT_NEAR(
				real(val1[i] * val2),
				real(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			ASSERT_NEAR(
				imag(val1[i] * val2),
				imag(val3[i]),
				ERRBOUND)  << "Fail for the " << i << "-th coefficient";
			diff += abs((val1[i] * val2) - (val3[i]));
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



TEST_P(TestCKKSBasicHomomorphism, DivByConstant)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(p.pt)/2), sqrt(get_range(p.pt)/2));
	complex<double> diff = 0;
	int slots = CUDAEngine::N;
	complex<double> *val1 = new complex<double>[slots];

	for(int N = 0; N < NTESTSINTENSE; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val1[i] = distribution(generator);
		double val2 = distribution(generator);
		val2 += 0.1 * (abs(1.0 / val2) >= get_range(p.pt)/2);

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
		complex<double> *val3 = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSBasicHomomorphism, RotateRight)
{      
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt)/2, get_range(p.pt)/2);
	int slots = p.nphi/2;
	complex<double> diff = 0;

    CKKSContext *cipher = m_cipher;

	//////////////
	// Sampling //
	//////////////
	complex<double> *val1 = new complex<double>[slots];
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
		complex<double>* val3 = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSBasicHomomorphism, RotateLeft)
{      
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt)/2, get_range(p.pt)/2);
	int slots = p.nphi/2;
	complex<double> diff = 0;

    CKKSContext *cipher = m_cipher;

	//////////////
	// Sampling //
	//////////////
	complex<double> *val1 = new complex<double>[slots];
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
		complex<double>* val3 = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, Conjugate)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), distribution(generator)};

		// std::cout << "Will compute " << val << " ^ " << x << std::endl;
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
		// cipher_t *ct2 = ckks_conjugate(cipher, ct1);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++){
			// std::cout << "conj(" << val[i] << ") =" << conj(val[i]) << " =? " << result[i] << std::endl;
			ASSERT_NEAR(real(conj(val[i])), real(result[i]), ERRBOUND)  << "Fail for the " << i << "-th slot at test " << N;
			ASSERT_NEAR(imag(conj(val[i])), imag(result[i]), ERRBOUND)  << "Fail for the " << i << "-th slot at test " << N;
		}

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}
	delete[] val;
}

TEST_P(TestCKKSAdvHomomorphism, MulSequence)
{        
	int supported_depth = (p.k - 1);
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-1, 1);
	int slots = CUDAEngine::N;

    CKKSContext *cipher = m_cipher;
	complex<double> *expected = new complex<double>[slots];
	complex<double> *val1 = new complex<double>[slots];
	complex<double> *val2 = new complex<double>[slots];
	for(int N = 0; N < NTESTS; N++){

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
		ckks_decrypt(cipher, expected, ct2, sk);
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

TEST_P(TestCKKSAdvHomomorphism, InnerProductSingle)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-sqrt(get_range(p.pt)/2), sqrt(get_range(p.pt)/2));
	int length = 4;

	for(int N = 0; N < 1; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		complex<double> *val1 = new complex<double>[length];
		complex<double> *val2 = new complex<double>[length];
		complex<double> val;
		do{
			for(int i = 0; i < length; i++){
				val1[i] = {distribution(generator), distribution(generator)};
				val2[i] = {distribution(generator), distribution(generator)};
			}
			val = std::inner_product(
			&val1[0],
			&val1[length],
			&val2[0],
			(complex<double>){0.0, 0.0});
		} while(
			real(val) <= -get_range(p.pt) || real(val) >= get_range(p.pt) ||
			imag(val) <= -get_range(p.pt) || imag(val) >= get_range(p.pt));
		// std::cout << "Result expected: " << val << std::endl;
		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = new cipher_t[length];
		cipher_t *ct2 = new cipher_t[length];
		for(int i = 0; i < length; i++){
			ckks_encrypt(cipher, &ct1[i], &val1[i]);
			ckks_encrypt(cipher, &ct2[i], &val2[i]);
		}

		//////////////////
		// InnerProduct //
		//////////////////
		// CT
		cipher_t ct3;
		cipher->sk = sk;
		ckks_inner_prod(cipher, &ct3, ct1, ct2, length);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *val3 = ckks_decrypt(cipher, &ct3, sk);

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

TEST_P(TestCKKSAdvHomomorphism, InnerProductBatch)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-1, 1);
	int slots = CUDAEngine::N;
	complex<double> *val1 = new complex<double>[slots];
	complex<double> *val2 = new complex<double>[slots];

	for(int N = 0; N < 1; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		complex<double> val;
		do{
			for(int i = 0; i < slots; i++){
				val1[i] = {distribution(generator), distribution(generator)};
				val2[i] = {distribution(generator), distribution(generator)};
			}
			val = std::inner_product(
			&val1[0],
			&val1[slots],
			&val2[0],
			(complex<double>){0.0, 0.0});
		} while(
			real(val) <= -get_range(p.pt) || real(val) >= get_range(p.pt) ||
			imag(val) <= -get_range(p.pt) || imag(val) >= get_range(p.pt));
		
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
		cipher->sk = sk;
		
		ckks_batch_inner_prod(cipher, &ct3, ct1, ct2);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *val3 = ckks_decrypt(cipher, &ct3, sk);

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

TEST_P(TestCKKSAdvHomomorphism, Power)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	int powerbound = 5;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(
			-pow(get_range(p.pt), 1.0/(powerbound + 1)), pow(get_range(p.pt), 1.0/(powerbound + 1))
			);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, PowerOf2)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		int x = (1 << (rand() % 4));
		std::uniform_real_distribution<double> distribution = 
			std::uniform_real_distribution<double>(-pow(get_range(p.pt), 1.0/(x*2)), pow(get_range(p.pt), 1.0/(x*2)));

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, evalpoly)
{

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	int polydegree = 11;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-1, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

		//////////////
		// Sampling //
		//////////////
		for(int i = 0; i < slots; i++)
			val[i] = {distribution(generator), 0};
		double *coeffs = new double[polydegree];
		for(int i = 0; i < polydegree; i++)
			coeffs[i] = distribution(generator);

		/////////////
		// Encrypt //
		/////////////
		cipher_t *ct1 = ckks_encrypt(cipher, val, slots);

		////////////
		// Square //
		////////////
		cipher->sk = sk;
		cipher_t *ct2 = ckks_eval_polynomial(cipher, ct1, coeffs, polydegree);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int j = 0; j < slots; j++){
			complex<double> expected = 0;
			for(int i = polydegree-1; i >= 0; i--)
				expected = coeffs[i] + val[j] * expected;
			ASSERT_NEAR(real(expected), real(result[j]), ERRBOUND) << ": couldn't eval at "<< val << " for polydegree " << polydegree;
			ASSERT_NEAR(imag(expected), imag(result[j]), ERRBOUND) << ": couldn't eval at "<< val << " for polydegree " << polydegree;
		}
		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}
}

TEST_P(TestCKKSAdvHomomorphism, exp)
{

	if(p.pt <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, sin)
{
	if(p.pt <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];
	
	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, cos)
{
	if(p.pt <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];
	
	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, sigmoid)
{
	if(p.pt <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];

	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, log)
{
	if(p.pt <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0.1, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];
	
	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;
        cipher->sk = sk;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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
        CKKSContext *cipher = m_cipher;

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
		cipher_t *ct2 = ckks_log(cipher, ct1, CUDAEngine::get_n_residues(QBase)-1);

		//////////////
		// Decrypt  //
		//////////////
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, lnminus)
{
	if(p.pt <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];
	
	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;
        cipher->sk = sk;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		for(int i = 0; i < slots; i++)
			ASSERT_NEAR(log(1-real(val[i])), real(result[i]), ERRBOUND) << " Failure at slot " << i << " at test " << N << " log(1-"<< real(val[i])<<")";

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);
	}

	float x = 0;
	do{
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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

TEST_P(TestCKKSAdvHomomorphism, Inverse)
{
	if(p.pt <= 52)
		GTEST_SKIP_("Precision too low");

	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(0.1, 1);
	int slots = CUDAEngine::N;
	complex<double> *val = new complex<double>[slots];
	
	for(int N = 0; N < NTESTS; N++){
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

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
        CKKSContext *cipher = m_cipher;

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
		complex<double> *result = ckks_decrypt(cipher, ct2, sk);

		///////////
		// Check //
		///////////
		std::cout << "Computing 1.0 / " << (real(val[0])) << std::endl;
		std::cout << "Expected: " << 1.0 / (real(val[0])) << std::endl;
		std::cout << "Received: " << real(result[0]) << std::endl;
		if(abs(1.0 / (real(val[0]))  -  real(result[0])) > 0.1)
				break;

		cipher_free(cipher, ct1);
		cipher_free(cipher, ct2);
		free(result);

		x += 0.05;
	}while(1);
	std::cout << "Could validate the inverse in the range [0," << x << ")" << std::endl;
}


// TEST_P(TestCKKSBasics, ImportExportKeys)
// {
//     CKKSContext *cipher = m_cipher;
// 	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
// 	std::default_random_engine generator;
// 	std::uniform_real_distribution<double> distribution = 
// 		std::uniform_real_distribution<double>(-get_range(p.pt), get_range(p.pt));

// 	cipher->sk = sk;
// 	for(int N = 0; N < NTESTS; N++){
// 		// Export
// 		json jkeys = cipher->export_keys();

// 		// Encrypt something
// 		complex<double> val = {distribution(generator), distribution(generator)};

// 		cipher_t* ct = ckks_encrypt(cipher, &val);

// 		// Clear keys
// 		cipher->clear_keys();

// 		// Import
// 		cipher->load_keys(jkeys);

// 		// Verifies if we are still able to decrypt
// 		complex<double> *val_decoded = ckks_decrypt(cipher, ct, sk);

// 		ASSERT_NEAR(real(val), real(*val_decoded), ERRBOUND);
// 		ASSERT_NEAR(imag(val), imag(*val_decoded), ERRBOUND);

// 		delete[] val_decoded;
// 		cipher_free(cipher, ct);
// 	}
// }


TEST_P(TestCKKSBasics, ImportExportCiphertext)
{
    CKKSContext *cipher = m_cipher;
	// Produces random floating-point values i, uniformly distributed on the interval [a, b)
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution = 
		std::uniform_real_distribution<double>(-get_range(p.pt), get_range(p.pt));

	for(int N = 0; N < NTESTS; N++){
		// Encrypt something
		complex<double> val = {distribution(generator), distribution(generator)};

		cipher_t* ct = ckks_encrypt(cipher, &val);

		// Export
		json exported_ct = cipher_export(cipher, ct);

		// StringBuffer buffer;
		// buffer.Clear();
		// PrettyWriter<StringBuffer> writer(buffer);
		// exported_ct.Accept(writer);
		// std::cout << (buffer.GetString()) << std::endl;

		// Import
		cipher_t* imported_ct = cipher_import(cipher, exported_ct);	

		// Verifies if we are still able to decrypt
		complex<double> *val_decoded = ckks_decrypt(cipher, imported_ct, sk);

		ASSERT_NEAR(real(val), real(*val_decoded), ERRBOUND);
		ASSERT_NEAR(imag(val), imag(*val_decoded), ERRBOUND);

		delete[] val_decoded;
		cipher_free(cipher, ct);
		cipher_free(cipher, imported_ct);
	}
}

//
//Defines for which parameters set AOADGT will be tested.
//It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<TestParams> basicparams = ::testing::Values(
	//{   nphi, k, kl, scalingfactor},
	(TestParams){128, 2, 3, 55},
	(TestParams){512, 2, 3, 55},
	(TestParams){2048, 2, 3, 55},
	(TestParams){4096, 2, 3, 55},
	(TestParams){8192, 2, 3, 55},
	(TestParams){16384, 2, 3, 55},
	(TestParams){32768, 2, 3, 55},
	(TestParams){128, 5, 5, 55},
	(TestParams){2048, 5, 5, 55},
	(TestParams){4096, 5, 5, 55},
	(TestParams){8192, 5, 5, 55},
	(TestParams){16384, 5, 5, 55},
	(TestParams){32768, 5, 5, 55},
	(TestParams){128, 10, 11, 55},
	(TestParams){2048, 10, 11, 55},
	(TestParams){4096, 10, 11, 55},
	(TestParams){8192, 10, 11, 55},
	(TestParams){16384, 10, 11, 55},
	(TestParams){32768, 10, 11, 55}
	);
::testing::internal::ParamGenerator<TestParams> advparams = ::testing::Values(
	//{   nphi, k, kl, scalingfactor},
	// (TestParams){128, 4, 5, 45},
	// (TestParams){128, 30, 31, 48},
	(TestParams){128, 20, 21, 55},
	(TestParams){128, 30, 31, 55},
	(TestParams){2048, 20, 21, 55},
	(TestParams){2048, 30, 31, 55},
	(TestParams){2048, 40, 41, 55},
	(TestParams){4096, 20, 21, 55},
	(TestParams){4096, 30, 31, 55},
	(TestParams){8192, 20, 21, 55},
	(TestParams){8192, 30, 31, 55},
	(TestParams){16384, 20, 21, 55},
	(TestParams){16384, 30, 31, 55},
	(TestParams){32768, 20, 21, 55},
	(TestParams){32768, 30, 31, 55}
	);
std::string printParamName(::testing::TestParamInfo<TestParams> p){
	TestParams params = p.param;

	return std::to_string(params.nphi) +
	"_k" + std::to_string(params.k) + "_kl" + std::to_string(params.kl) + 
	"_factor" + std::to_string(params.scalingfactor);
}

INSTANTIATE_TEST_CASE_P(AOADGT,
	TestCKKSBasics,
	basicparams,
	printParamName
);

INSTANTIATE_TEST_CASE_P(AOADGT,
	TestCKKSBasicHomomorphism,
	basicparams,
	printParamName
);

INSTANTIATE_TEST_CASE_P(AOADGT,
	TestCKKSAdvHomomorphism,
	advparams,
	printParamName
);


int main(int argc, char **argv) {
  //////////////////////////
  ////////// Google tests //
  //////////////////////////
  std::cout << "Testing AOADGT " << GET_AOADGT_VERSION() << std::endl;
  std::cout << "Running " << NTESTS << std::endl << std::endl;
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
