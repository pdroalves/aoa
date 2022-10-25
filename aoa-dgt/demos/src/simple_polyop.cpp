// cuPoly - A GPGPU-based library for doing polynomial arithmetic on RLWE-based cryptosystems
// Copyright (C) 2017-2021, Pedro G. M. R. Alves - pedro.alves@ic.unicamp.br

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


    
#include <cuda_runtime_api.h>
#include <NTL/ZZ.h>
#include <time.h>
#include <unistd.h>
#include <cuPoly/settings.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <cuPoly/cuda/sampler.h>


/**
  simple_polyop:
  
  Here we demonstrate how you can initialize two polynomials, execute some 
  operations, and print the outcome.
  
 */
 int main(){
  // Sets the logger to QUIET
  Logger::getInstance()->set_mode(QUIET);

  // Params
  int nphi = 32;
  int t = 256;
  int k = 10;

  // Init the generic engine
  CUDAEngine::init(k, k + 1, nphi, t);// Init CUDA
  
  // The context store relevant pre-computed values
  Context *ctx = new Context();

  // Init the sampler engine
  Sampler::init(ctx);

  /////////////
  // Message //
  /////////////
  poly_t m1, m2;
  poly_init(ctx, &m1);
  poly_init(ctx, &m2);

  // Different initializations
  Sampler::sample(ctx, &m1, UNIFORM);
  poly_set_coeff(ctx, &m2, 0, to_ZZ(1));
  poly_set_coeff(ctx, &m2, 3, to_ZZ(42));

  ////////////////
  // Operations //
  ///////////////

  // You don't need to explictly initialize a poly_t.
  // It is enough to declare it and let it to be initialized inside some 
  // function.
  poly_t m3;
  poly_mul(ctx, &m3, &m1, &m2);

  // However, sometimes you may prefer to do that in advance.
  poly_t m4;
  poly_init(ctx, &m4);
  poly_add(ctx, &m4, &m1, &m2);

  ///////////
  // Print //
  ///////////
  std::cout << "m3: " << poly_to_string(ctx, &m3) << std::endl;
  std::cout << "m4: " << poly_to_string(ctx, &m4) << std::endl;
  std::cout << "The 4th coefficient of m2 is: " << poly_get_coeff(ctx, &m2, 3) << std::endl;
  
  /////////////
  // Release //
  /////////////
  poly_free(ctx, &m1);
  poly_free(ctx, &m2);
  poly_free(ctx, &m3);
  poly_free(ctx, &m4);

  CUDAEngine::destroy();
  cudaDeviceReset();
  cudaCheckError();
  return 0;
}
