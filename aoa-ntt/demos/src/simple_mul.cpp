#include <newckks/ckks/ckks.h>
#include <newckks/ckks/ckkscontext.h>
#include <newckks/ckks/ckkskeys.h>
#include <newckks/arithmetic/poly_t.h>
#include <newckks/cuda/manager.h>
#include <newckks/tool/version.h>
#include <newckks/cuda/sampler.h>
#include <newckks/ckks/encoder.h>
#include <stdlib.h>
#include <NTL/ZZ.h>
#include <cuda_profiler_api.h>
#include <random>
#include <sstream>
#include <string>

NTL_CLIENT

int main() {
  cudaProfilerStop();

  ZZ q;

  srand(0);
  NTL::SetSeed(to_ZZ(0));

  // Params
  int nphi = 4096;
  int k = 2;
  int kl = 2;
  int scalingfactor = 55;
  int slots = nphi/2;

  // Init
  std::cout << "Scaling factor: " << scalingfactor << std::endl;
  CUDAManager::init(k, kl, nphi, scalingfactor);// Init CUDA
        
  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  /////////////
  // Message //
  /////////////
  std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0, 5);
  std::default_random_engine generator;

  /////////////
  // Encrypt //
  /////////////
  std::cout << "Will encrypt" <<std::endl;

  std::complex<double> *val1 = new std::complex<double>[slots];
  for(int i = 0; i < slots; i++)
    val1[i] = {distribution(generator), distribution(generator)};
  double val2 = 2;

  cipher_t* ct1 = ckks_encrypt(cipher, val1, slots);
  cipher_t* ct2 = ckks_encrypt(cipher, val2);

  //////////
  // Mul //
  ////////
  cipher_t *ct3 = new cipher_t;
  cipher_init(cipher, ct3);
  ckks_mul(cipher, ct3, ct1, val2);

  complex<double> *m3 = ckks_decrypt(cipher, ct3, sk);
  for(int i = 0; i < slots; i++){
    complex<double> expected = val1[i] *  val2;
    std::cout << val1[i] << " * " << val2 << "\t\t == " << m3[i] << " Expected: " <<  expected << std::endl;
    assert(abs(real(val1[i] * val2) - real(m3[i])) < 1e-3);
    assert(abs(imag(val1[i] * val2) - imag(m3[i])) < 1e-3);
  }
  
  cipher_free(cipher, ct1);
  cipher_free(cipher, ct2);
  cipher_free(cipher, ct3);
  delete[] val1;
  delete ct1;
  delete ct2;
  delete ct3;
  ckkskeychain_free(cipher, keys);
  delete keys;
  ckks_free_sk(cipher, sk);
  delete sk;
  delete cipher;
  CUDAManager::destroy();
  cudaDeviceReset();
  cudaCheckError();
  return 0;
}