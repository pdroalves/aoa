#ifndef CONTEXT_H
#define CONTEXT_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <newckks/tool/context.h>
#include <newckks/defines.h>

class Sampler;

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
    Sampler *s;

  public:
    uint64_t *h_tmp; // Auxiliar for encoding
    uint64_t *d_tmp; // Auxiliar for rotations

    GaussianInteger *d_gi_a, *d_gi_b, *d_gi_c, *d_gi_d; // Conversion to and from GaussianIntegers
    
  Context();

  ~Context();

  /*! \brief Returns a reference for the cudaStream_t related to the context
  *
  */
  cudaStream_t get_stream(){
    return stream;
  };

  Sampler* get_sampler(){
    return s;
  };


};

#endif