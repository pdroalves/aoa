#ifndef CONTEXT_H
#define CONTEXT_H
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdexcept>
#include <cstring>
#include <NTL/ZZ.h>
#include <AOADGT/settings.h>
#include <AOADGT/arithmetic/polynomial.h>
#include <AOADGT/cuda/cudaengine.h>

/**
 * @brief      Implements the Context design pattern. Operations happening
 *             between elements in the same context are not thread safe.
 */
class Context{
  private:
    /*! \brief Defines the stream that should be used on the context
    *
    * It is expected that operations applied over elements in the same context 
    * will always happen in a serial way. This way, each context follows the 
    * same cudaStream_t.
    */
    cudaStream_t stream;

  public:
    // RNS
    /*! \brief RNS coefficients on host
    *
    * Stores a copy of residues coefficients on host memory
    */
    GaussianInteger *h_coefs;

    /*! \brief Auxiliar array to be used on poly_copy_to_host, poly_sub, poly_basis_extension_B_to_Q
    *
    * In poly_copy_host, before copying to the host memory, we need to apply the IDGT on 
    * the polynomial. Since we do not want to mess with coefficients on the 
    * device memory, they are temporarily copied to d_aux_coefs.
     * 
    */
    GaussianInteger *d_aux_coefs;

    // HPS
    uint64_t *d_coefs_B; //!< Temporary array to be used on poly_complex_scaling_tDivQ
    uint64_t *d_v; //!< Temporary array to be used on CUDAEngine::execute_polynomial_basis_ext_Q_to_B
    uint64_t *d_aux; //!< Temporary array to be used on CUDAEngine::execute_polynomial_basis_ext_Q_to_B and Sampler::call_get_binary_hweight_sample

    uint64_t *h_aux; //!< Temporary array to be used on Sampler::call_get_binary_hweight_sample

    // poly_xi
    GaussianInteger **h_b; //!< Temporary array to be used on poly_xi
    GaussianInteger **d_b; //!< Temporary array to be used on poly_xi
    
    // dgt
    /*! \brief Auxiliar array to be used on runDGTKernelHierarchical
    *
    * Because of its design, hdgt_tr cannot have its outcome written to the
    * input array. For this reason, we use this variable to temporarily store the 
    * outcome.
    */
    GaussianInteger *d_tmp_data;

    /*! \brief Returns a reference for the cudaStream_t related to the context
    *
    */
    cudaStream_t get_stream(){
      return stream;
    }
    
    Context();

    ~Context(){
      cudaDeviceSynchronize();
      cudaCheckError();

      free(h_b);
      cudaFree(d_b);
      cudaCheckError();

      cudaStreamDestroy(stream);
      cudaCheckError();
    
      free(h_aux);
      free(h_coefs);
      cudaFree(d_aux_coefs);
      cudaFree(d_v);
      cudaFree(d_aux);
      cudaFree(d_coefs_B);
      cudaFree(d_tmp_data);
    }
};

#endif
