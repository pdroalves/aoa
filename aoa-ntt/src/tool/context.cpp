#include <newckks/tool/context.h>
#include <newckks/arithmetic/poly_t.h>
#include <newckks/cuda/sampler.h>

Context::Context(){
	cudaStreamCreate(&stream);
	cudaCheckError();

    h_tmp = (uint64_t*) malloc (poly_get_size(QBBase));
	cudaMalloc((void**)&d_tmp, poly_get_size(QBBase));
	cudaCheckError();

    // Sampler
    s = new Sampler(this);

    // DGT
    cudaMalloc((void**)&d_gi_a, poly_get_size(QBBase));
    cudaCheckError();
    cudaMalloc((void**)&d_gi_b, poly_get_size(QBBase));
    cudaCheckError();
    cudaMalloc((void**)&d_gi_c, poly_get_size(QBBase));
    cudaCheckError();
    cudaMalloc((void**)&d_gi_d, poly_get_size(QBBase));
    cudaCheckError();
};

Context::~Context(){
    cudaDeviceSynchronize();
    cudaCheckError();

    delete s;

    free(h_tmp);
    cudaFree(d_tmp);
    cudaCheckError();

    cudaFree(d_gi_a);
    cudaCheckError();
    cudaFree(d_gi_b);
    cudaCheckError();
    cudaFree(d_gi_c);
    cudaCheckError();
    cudaFree(d_gi_d);
    cudaCheckError();

    cudaStreamDestroy(stream);
    cudaCheckError();
  };