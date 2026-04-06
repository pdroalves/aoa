#include <stdlib.h>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <cxxopts.hpp>
#include <cuda_profiler_api.h>
#include <newckks/ckks/ckks.h>
#include <newckks/ckks/ckkskeys.h>
#include <newckks/ckks/ckkscontext.h>
#include <newckks/tool/version.h>

using namespace NTL;

#define BILLION  1000000000L
#define MILLION  1000000L
#define NITERATIONS 100


__host__ double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

double print_memory_usage(){
  // show memory usage of GPU
  size_t free_byte ; size_t total_byte ;
  cudaMemGetInfo( &free_byte, &total_byte ) ;
  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;

  std::cout << "GPU memory usage: used = " << used_db/1024.0/1024.0 << 
  ", free = " << free_db/1024.0/1024.0 << " MB, total = " <<
  total_db/1024.0/1024.0 << " MB\n" << std::endl;

  return free_db/1024.0/1024.0;
}


double runAdd(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  cipher->keys.pk = ckks_new_pk(cipher, sk);

  std::complex<double> val1 = {distribution(generator), distribution(generator)};
  std::complex<double> val2 = {distribution(generator), distribution(generator)};

  std::vector<cipher_t*> ct1;
  std::vector<cipher_t*> ct2;
  std::vector<cipher_t*> ct3;
  
  for(int i = 0; i < NITERATIONS; i++){
    ct1.push_back(ckks_encrypt(cipher, &val1));
    ct2.push_back(ckks_encrypt(cipher, &val2));
    cipher_t *c = new cipher_t;
    cipher_init(cipher, c);
    ct3.push_back(c);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_add(cipher, ct3[i], ct1[i], ct2[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
      ckks_add(cipher, ct3[i], ct1[i], ct2[i]);  
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);
  cudaProfilerStop();

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_free(cipher, ct1[i]);
    cipher_free(cipher, ct2[i]);
    cipher_free(cipher, ct3[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete cipher;
  return latency / NITERATIONS;
}

__host__ double runPolyAddAdd(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  Context *ctx = new Context();
  std::vector<poly_t> a(NITERATIONS);
  std::vector<poly_t> b(NITERATIONS);
  std::vector<poly_t> c(NITERATIONS);
  std::vector<poly_t> d(NITERATIONS);
  std::vector<poly_t> e(NITERATIONS);
  std::vector<poly_t> f(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &a[i]);
    poly_init(ctx, &b[i]);
    poly_init(ctx, &c[i]);
    poly_init(ctx, &d[i]);
    poly_init(ctx, &e[i]);
    poly_init(ctx, &f[i]);

    ctx->get_sampler()->sample_DG(&a[i], QBase);
    ctx->get_sampler()->sample_DG(&b[i], QBase);
    ctx->get_sampler()->sample_DG(&c[i], QBase);
    ctx->get_sampler()->sample_DG(&d[i], QBase);
    ctx->get_sampler()->sample_DG(&e[i], QBase);
    ctx->get_sampler()->sample_DG(&f[i], QBase);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    poly_add_add(ctx, &a[0], &b[0], &c[0], &d[0], &e[0], &f[0]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    poly_add_add(ctx, &a[i], &b[i], &c[i], &d[i], &e[i], &f[i]);
  cudaEventRecord(stop, ctx->get_stream());

  //
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++){
    poly_free(ctx, &a[i]);
    poly_free(ctx, &b[i]);
    poly_free(ctx, &c[i]);
    poly_free(ctx, &d[i]);
    poly_free(ctx, &e[i]);
    poly_free(ctx, &f[i]);
  }
  delete ctx;
  return latency / NITERATIONS;
}

__host__ double runMul(){
  cudaProfilerStop();
  cudaCheckError();
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);
  cipher->keys.pk = ckks_new_pk(cipher, sk);
  cipher->keys.evk = ckks_new_mtk(cipher, sk);

  std::complex<double> val1 = {distribution(generator), distribution(generator)};
  std::complex<double> val2 = {distribution(generator), distribution(generator)};

  std::vector<cipher_t*> ct1;
  std::vector<cipher_t*> ct2;
  std::vector<cipher_t*> ct3;
  
  for(int i = 0; i < NITERATIONS; i++){
    ct1.push_back(ckks_encrypt(cipher, &val1));
    ct2.push_back(ckks_encrypt(cipher, &val2));
    cipher_t *c = new cipher_t;
    cipher_init(cipher, c);
    ct3.push_back(c);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_mul_without_rescale(cipher, ct3[i], ct1[i], ct2[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaCheckError();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ckks_mul_without_rescale(cipher, ct3[i], ct1[i], ct2[i]);  
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  cudaProfilerStop();
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_free(cipher, ct1[i]);
    cipher_free(cipher, ct2[i]);
    cipher_free(cipher, ct3[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}

__host__ double runRescale(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::complex<double> val = {distribution(generator), distribution(generator)};

  std::vector<cipher_t*> ct;
  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_t *c = ckks_encrypt(cipher, &val);
    
    COMMONEngine::execute( cipher, &c->c[0], INVERSE);
    COMMONEngine::execute( cipher, &c->c[1], INVERSE);

    ct.push_back(c);
  }

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ckks_rescale(cipher, ct[i]);
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++)
    cipher_free(cipher, ct[i]);
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}

__host__ double runRotate(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::complex<double> val = {distribution(generator), distribution(generator)};

  std::vector<cipher_t*> ct1;
  std::vector<cipher_t*> ct2;
  
  for(int i = 0; i < NITERATIONS; i++){
    ct1.push_back(ckks_encrypt(cipher, &val, 1, (CUDAManager::N>>1) - 1));
    cipher_t *c = new cipher_t;
    cipher_init(cipher, c);
    ct2.push_back(c);
  }


  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_rotate_left(cipher, ct2[i], ct1[i], 1);

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ckks_rotate_left(cipher, ct2[i], ct1[i], 1);
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_free(cipher, ct1[i]);
    cipher_free(cipher, ct2[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}

__host__ double runSumslots(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::complex<double> val = {distribution(generator), distribution(generator)};

  std::vector<cipher_t*> ct;
  
  for(int i = 0; i < NITERATIONS; i++){
    ct.push_back(ckks_encrypt(cipher, &val, 1, (CUDAManager::N>>1) - 1));
  }


  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_sumslots(cipher, ct[i], ct[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ckks_sumslots(cipher, ct[i], ct[i]);
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);
  cudaProfilerStop();

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_free(cipher, ct[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}

__host__ double runConjugate(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::complex<double> val = {distribution(generator), distribution(generator)};

  std::vector<cipher_t*> ct1;
  std::vector<cipher_t*> ct2;
  
  for(int i = 0; i < NITERATIONS; i++){
    ct1.push_back(ckks_encrypt(cipher, &val, 1, (CUDAManager::N>>1) - 1));
    cipher_t *c = new cipher_t;
    cipher_init(cipher, c);
    ct2.push_back(c);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_conjugate(cipher, ct2[i], ct1[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ckks_conjugate(cipher, ct2[i], ct1[i]);

  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_free(cipher, ct1[i]);
    cipher_free(cipher, ct2[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}

double runAddPlain(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::complex<double> val1 = {distribution(generator), distribution(generator)};
  double val2 = distribution(generator);

  std::vector<cipher_t*> ct1;
  std::vector<cipher_t*> ct3;
  
  for(int i = 0; i < NITERATIONS; i++){
    ct1.push_back(ckks_encrypt(cipher, &val1));
    cipher_t *c = new cipher_t;
    cipher_init(cipher, c);
    ct3.push_back(c);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_add(cipher, ct3[i], ct1[i], val2);

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
      ckks_add(cipher, ct3[i], ct1[i], val2);  
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_free(cipher, ct1[i]);
    cipher_free(cipher, ct3[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}



double runMulPlain(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::complex<double> val1 = {distribution(generator), distribution(generator)};
  double val2 = distribution(generator);

  std::vector<cipher_t*> ct1;
  std::vector<cipher_t*> ct3;
  
  for(int i = 0; i < NITERATIONS; i++){
    ct1.push_back(ckks_encrypt(cipher, &val1));
    cipher_t *c = new cipher_t;
    cipher_init(cipher, c);
    ct3.push_back(c);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_mul(cipher, ct3[i], ct1[i], val2);

  // Benchmark
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
      ckks_mul(cipher, ct3[i], ct1[i], val2);  
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    cipher_free(cipher, ct1[i]);
    cipher_free(cipher, ct3[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}

__host__ double runEnc(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::vector<poly_t*>   ms(NITERATIONS);
  std::vector<cipher_t> cts(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    ms[i] = new poly_t;
    poly_init(cipher, ms[i]);
    cipher_init(cipher, &cts[i]);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ckks_encrypt_poly(cipher, &cts[i], ms[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ckks_encrypt_poly(cipher, &cts[i], ms[i]);
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  // Release memory  
  for(int i = 0; i < NITERATIONS; i++){
    poly_free(cipher, ms[i]);
    cipher_free(cipher, &cts[i]);
  }
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}


__host__ double runDec(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;

  float latency = 0;

  // Setup
  CKKSContext *cipher = new CKKSContext();
  SecretKey *sk = ckks_new_sk(cipher);
  CKKSKeychain *keys = ckks_keygen(cipher, sk);

  std::complex<double> m = {distribution(generator), distribution(generator)};
  std::vector<poly_t*> pts;
  std::vector<cipher_t*> cts;
  
  for(int i = 0; i < NITERATIONS; i++){
    cts.push_back(ckks_encrypt(cipher, &m));
    poly_t *pt = new poly_t;
    poly_init(cipher, pt);
    pts.push_back(pt);
  }
  // Warm-up
  ckks_decrypt_poly(cipher, pts[0], cts[0], sk);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, cipher->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ckks_decrypt_poly(cipher, pts[i], cts[i], sk);
  cudaEventRecord(stop, cipher->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);


  // Release memory  
  for(std::vector<cipher_t*>::iterator it = cts.begin(); it != cts.end(); ++it)
    cipher_free(cipher, *it);
  for(std::vector<poly_t*>::iterator it = pts.begin(); it != pts.end(); ++it)
    poly_free(cipher, *it);
    
  ckks_free_sk(cipher, sk);
  delete sk;
  delete keys;
  delete cipher;
  return latency / NITERATIONS;
}

__host__ double runNTT(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  CKKSContext *ctx = new CKKSContext();
  std::vector<poly_t> d_data(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &d_data[i]);
    ctx->get_sampler()->sample_DG(&d_data[i], QBase);
  }

  // Benchmark
  cudaDeviceSynchronize();
  cudaEventRecord(start, ctx->get_stream());
  cudaProfilerStart();
  for(int i = 0; i < NITERATIONS; i++)
    COMMONEngine::execute( ctx, &d_data[i], FORWARD);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++)
    poly_free(ctx, &d_data[i]);
  delete ctx;
  return latency / NITERATIONS;
}

__host__ double runINTT(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  CKKSContext *ctx = new CKKSContext();
  std::vector<poly_t> d_data(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &d_data[i]);
    ctx->get_sampler()->sample_DG(&d_data[i], QBase);
    COMMONEngine::execute( ctx, &d_data[i], FORWARD);
  }

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    COMMONEngine::execute( ctx, &d_data[i], INVERSE);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();

  //
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++)
    poly_free(ctx, &d_data[i]);
  delete ctx;
  return latency / NITERATIONS;
}

__host__ double runPolyAdd(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  CKKSContext *ctx = new CKKSContext();
  std::vector<poly_t> a_data(NITERATIONS);
  std::vector<poly_t> b_data(NITERATIONS);
  std::vector<poly_t> c_data(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &a_data[i]);
    poly_init(ctx, &b_data[i]);
    poly_init(ctx, &c_data[i]);
    ctx->get_sampler()->sample_DG(&a_data[i], QBase);
    ctx->get_sampler()->sample_DG(&b_data[i], QBase);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    poly_add(ctx, &c_data[i], &a_data[i], &b_data[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    poly_add(ctx, &c_data[i], &a_data[i], &b_data[i]);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++){
    poly_free(ctx, &a_data[i]);
    poly_free(ctx, &b_data[i]);
    poly_free(ctx, &c_data[i]);
  }
  delete ctx;
  return latency / NITERATIONS;
}

__host__ double runPolyMul(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  CKKSContext *ctx = new CKKSContext();
  std::vector<poly_t> a_data(NITERATIONS);
  std::vector<poly_t> b_data(NITERATIONS);
  std::vector<poly_t> c_data(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &a_data[i]);
    poly_init(ctx, &b_data[i]);
    poly_init(ctx, &c_data[i]);
    ctx->get_sampler()->sample_DG(&a_data[i], QBase);
    ctx->get_sampler()->sample_DG(&b_data[i], QBase);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    poly_mul(ctx, &c_data[i], &a_data[i], &b_data[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    poly_mul(ctx, &c_data[i], &a_data[i], &b_data[i]);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++){
    poly_free(ctx, &a_data[i]);
    poly_free(ctx, &b_data[i]);
    poly_free(ctx, &c_data[i]);
  }
  delete ctx;
  return latency / NITERATIONS;
}

__host__ double runPolyMulAdd(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  CKKSContext *ctx = new CKKSContext();
  std::vector<poly_t> a_data(NITERATIONS);
  std::vector<poly_t> b_data(NITERATIONS);
  std::vector<poly_t> c_data(NITERATIONS);
  std::vector<poly_t> d_data(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &a_data[i]);
    poly_init(ctx, &b_data[i]);
    poly_init(ctx, &c_data[i]);
    poly_init(ctx, &d_data[i]);
    ctx->get_sampler()->sample_DG(&a_data[i], QBase);
    ctx->get_sampler()->sample_DG(&b_data[i], QBase);
    ctx->get_sampler()->sample_DG(&c_data[i], QBase);
  }

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    poly_mul_add(ctx, &d_data[i], &a_data[i], &b_data[i], &c_data[i]);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    poly_mul_add(ctx, &d_data[i], &a_data[i], &b_data[i], &c_data[i]);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++){
    poly_free(ctx, &a_data[i]);
    poly_free(ctx, &b_data[i]);
    poly_free(ctx, &c_data[i]);
    poly_free(ctx, &d_data[i]);
  }
  delete ctx;
  return latency / NITERATIONS;
}


__host__ double runDiscreteGaussianSampler(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  Context *ctx = new Context();
  std::vector<poly_t> a(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++)
    poly_init(ctx, &a[i]);

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ctx->get_sampler()->sample_DG(&a[i], QBase);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ctx->get_sampler()->sample_DG(&a[i], QBase);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++)
    poly_free(ctx, &a[i]);
  delete ctx;
  return latency / NITERATIONS;
 }

__host__ double runNarrowSampler(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  Context *ctx = new Context();
  std::vector<poly_t> a(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++)
    poly_init(ctx, &a[i]);

  // Warm-up
  for(int i = 0; i < NITERATIONS; i++)
    ctx->get_sampler()->sample_DG(&a[i], QBase);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    ctx->get_sampler()->sample_DG(&a[i], QBase);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++)
    poly_free(ctx, &a[i]);
  delete ctx;
  return latency / NITERATIONS;
 }

__host__ double runModUp(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  Context *ctx = new Context();
  std::vector<poly_t> a(NITERATIONS);
  std::vector<poly_t> b(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &a[i]);
    poly_init(ctx, &b[i], QBBase);
    ctx->get_sampler()->sample_DG(&a[i], QBase);
  }

  // Warm-up
  poly_modup(ctx, &b[0], &a[0], CUDAManager::RNSQPrimes.size() - 1);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    poly_modup(ctx, &b[i], &a[i], CUDAManager::RNSQPrimes.size() - 1);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++){
    poly_free(ctx, &a[i]);
    poly_free(ctx, &b[i]);
  }
  delete ctx;
  return latency / NITERATIONS;
}

__host__ double runModDown(){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float latency = 0;

  // Setup
  Context *ctx = new Context();
  std::vector<poly_t> a(NITERATIONS);
  std::vector<poly_t> b(NITERATIONS);
  for(int i = 0; i < NITERATIONS; i++){
    poly_init(ctx, &a[i], QBBase);
    poly_init(ctx, &b[i]);
    ctx->get_sampler()->sample_DG(&a[i], QBBase);
  }

  // Warm-up
  poly_moddown(ctx, &b[0], &a[0], CUDAManager::RNSQPrimes.size() - 1);

  // Benchmark
  cudaDeviceSynchronize();
  cudaProfilerStart();
  cudaEventRecord(start, ctx->get_stream());
  for(int i = 0; i < NITERATIONS; i++)
    poly_moddown(ctx, &b[i], &a[i], CUDAManager::RNSQPrimes.size() - 1);
  cudaEventRecord(stop, ctx->get_stream());
  cudaCheckError();
  cudaProfilerStop();
  cudaCheckError();
  cudaEventSynchronize(stop);
  cudaCheckError();

  //
  cudaEventSynchronize(stop);
  cudaCheckError();
  cudaEventElapsedTime(&latency, start, stop);

  for(int i = 0; i < NITERATIONS; i++){
    poly_free(ctx, &a[i]);
    poly_free(ctx, &b[i]);
  }
  delete ctx;
  return latency / NITERATIONS;
}

template <class T>
std::vector<T> intersection(std::vector<T> &v1,
                                      std::vector<T> &v2){
    std::vector<T> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(),v1.end(),
                          v2.begin(),v2.end(),
                          back_inserter(v3));
    return v3;
}

 int main(int argc, char* argv[]){
  std::cout << "Benchmark " << GET_VERSION() << std::endl;
  #ifndef OPTMODUP
  std::cout << "Running the non-optimized version of ModUp." << std::endl;
  #endif
  cudaProfilerStop();

  /////////////////////////
  // Command line parser //
  ////////////////////////
  cxxopts::Options options("newckks_benchmark", "This program benchmarks the main procedures of newckks");
  options.add_options()
  ("p,procedure", "define which procedure should be measured", cxxopts::value<std::vector<std::string>>())
  ("d,degree", "for which ring degree it should measure the latencies", cxxopts::value<std::vector<int>>())
  ("g,gpu", "which parameter set should be used", cxxopts::value<std::string>())
  ("r,residues", "number of residues", cxxopts::value<int>())
  ("h,help", "Print help and exit.")
  ;
  auto result = options.parse(argc, argv);

  // help
  if (result.count("help")) {
    std::cout << options.help({""}) << std::endl;
    exit(0);
  }

  srand(0);

  // Output precision
  std::cout << std::fixed;
  std::cout.precision(4);    

  std::vector<std::string> type_data = {
    "PolyAdd",
    "PolyMul",
    "PolyMulAdd",
    "PolyDoubleAdd",
    "Encrypt",
    "Decrypt",
    "Add",
    "Mul",
    "AddPlain",
    "MulPlain",
    "Rescale",
    "Rotate",
    "Sumslots",
    "NTT", "INTT",
    "ModUp", "ModDown"
  };
  std::vector<int> degrees_data = {
    2048,
    4096,
    8192,
    16384,
    32768,
    65536
  };
  std::vector<std::vector<double>> data(type_data.size(), std::vector<double>());

  // Select procedures
  if (result.count("procedure")){
    std::vector<std::string> v = result["procedure"].as<std::vector<std::string>>();
    type_data = intersection(type_data, v);
  }

  // Select ring degrees
  if (result.count("degree")){
    std::vector<int> v = result["degree"].as<std::vector<int>>();
    degrees_data = intersection(degrees_data, v);
  }

  std::map<int, int> parameters;
  if(result.count("gpu")){
    if(result["gpu"].as<std::string>().compare("k80") == 0){
      parameters[2048] = 2;
      parameters[4096] = 3;
      parameters[8192] = 6;
      parameters[16384] = 12;
      parameters[32768] = 10;
      parameters[65536] = 10;
    }else if(result["gpu"].as<std::string>().compare("v100") == 0){
      parameters[2048] = 2;
      parameters[4096] = 2;
      parameters[8192] = 2;
      parameters[16384] = 6;
      parameters[32768] = 10;
      parameters[65536] = 10;
    }
  }

  if(parameters.size() == 0){
    // Load k80 parameters
      int default_k;
      if(result.count("residues"))
        default_k = result["residues"].as<int>();
      else
        default_k = 2;
      parameters[2048] = default_k;
      parameters[4096] = default_k;
      parameters[8192] = default_k;
      parameters[16384] = default_k;
      parameters[32768] = default_k;
      parameters[65536] = default_k;
  }
  float latency = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(std::vector<int>::iterator d = degrees_data.begin(); d != degrees_data.end(); d++){
    // Init
    // 
    // Values:
    // 
    int k, kl, nphi, scalingfactor;
    k = parameters[*d];
    kl = k+1;
    nphi = (*d);
    scalingfactor = 55;
    
    std::cout << k << ", " << kl << ", " << nphi << "< " << scalingfactor << std::endl;
    
    // Start the engine
    cudaEventRecord(start);
    CUDAManager::init(k, kl, nphi, scalingfactor);
    cudaEventRecord(stop);
    cudaCheckError();
    cudaEventSynchronize(stop);
    cudaCheckError();
    cudaEventElapsedTime(&latency, start, stop);
    std::cout << "Initialization done in " << latency << std::endl;
    
    ZZ_p::init(CUDAManager::get_q_product());

    std::cout << *d << " (" << NTL::NumBits(CUDAManager::get_q_product()) << " bits)" << std::endl;
    print_memory_usage();
    // Tests
    for(std::vector<std::string>::iterator it = type_data.begin(); it != type_data.end(); it++){
      std::cout << (*it) << std::endl;
      double diff;
      if(*it == "Encrypt")
        diff = runEnc();
      else if (*it == "Decrypt")
        diff = runDec();
      else if (*it == "Add")
        diff = runAdd();
      else if (*it == "Mul" && k >= 2)
        diff = runMul();
      else if (*it == "AddPlain")
        diff = runAddPlain();
      else if (*it == "MulPlain")
        diff = runMulPlain();
      else if (*it == "Rescale" && k >= 2)
        diff = runRescale();
      else if (*it == "Rotate")
        diff = runRotate();
      else if (*it == "Sumslots")
        diff = runSumslots();
      else if (*it == "Conjugate")
        diff = runConjugate();
      else if (*it == "NTT")
        diff = runNTT();
      else if (*it == "INTT")
        diff = runINTT();
      else if (*it == "PolyAdd")
        diff = runPolyAdd();
      else if (*it == "PolyDoubleAdd")
        diff = runPolyAddAdd();
      else if (*it == "PolyMul")
        diff = runPolyMul();
      else if (*it == "PolyMulAdd")
        diff = runPolyMulAdd();
      else if (*it == "DiscreteGaussianSampler")
        diff = runDiscreteGaussianSampler();
      else if (*it == "NarrowSampler")
        diff = runNarrowSampler();
      else if (*it == "ModUp")
        diff = runModUp();
      else if (*it == "ModDown")
        diff = runModDown();
      else 
        continue;

      data[distance(type_data.begin(), it)].push_back(diff);
    }

    // Release
    cudaDeviceSynchronize();
    cudaCheckError();

    CUDAManager::destroy();
    cudaCheckError();
  }
  
  // 
  // Output
  // 
  const char separator    = ' ';
  const int nameWidth     = 10;
  const int numWidth      = 30;

  // Print degrees
  std::cout << std::left << std::setw(numWidth) << "Placeholder" << std::setw(nameWidth) << std::setfill(separator);
  for(std::vector<int>::iterator d = degrees_data.begin(); d != degrees_data.end(); d++)
    std::cout << (*d) << std::setw(nameWidth) << std::setfill(separator);
  std::cout << std::endl;

  // Print data
  for(std::vector<std::string>::iterator it = type_data.begin(); it != type_data.end(); it++){
    // Type name
    std::cout << std::left << std::setw(numWidth) << (*it) << std::setw(nameWidth) << std::setfill(separator);

    // Values
    std::vector<double> v = data[distance(type_data.begin(), it)];
    for(std::vector<double>::iterator t = v.begin(); t != v.end(); t++)
      std::cout << (*t) << std::setw(nameWidth) << std::setfill(separator);
    std::cout << std::endl;
  }

  cudaDeviceReset();
  cudaCheckError();
}
