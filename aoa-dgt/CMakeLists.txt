PROJECT (AOADGT LANGUAGES CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

SET(LIB_NAME "AOADGT")
SET(PROJECT_VERSION_MAJOR "0")
SET(PROJECT_VERSION_MINOR "0")
SET(PROJECT_VERSION_PATCH "1")
SET(PROJECT_VERSION_TWEAK "0")
SET(AOADGT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${PROJECT_VERSION_TWEAK})")
SET(VERSION ${AOADGT_VERSION})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("\n-- Configuring ${PROJECT_NAME} ${AOADGT_VERSION} (${CMAKE_BUILD_TYPE})...\n")

SET(CMAKE_CXX_FLAGS "-Wall -Wextra")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CUDA_NVCC_FLAGS_DEBUG "-G")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/AOADGTConfig.h.in"
  "${PROJECT_SOURCE_DIR}/include/AOADGT/AOADGTConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find AOADGTConfig.h
include_directories("${PROJECT_BINARY_DIR}")

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
SET(GCC_CUDA_VERSION "gcc" CACHE STRING "GCC compiler to be used with cuda")
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

SET(CUDA_SEPARABLE_COMPILATION ON)

ENABLE_LANGUAGE(CUDA)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(NTL 11.0 REQUIRED)
FIND_PACKAGE(GMP 6.0)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(cxxopts REQUIRED)
FIND_PACKAGE(Doxygen)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

LIST(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)
LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread)
LIST(APPEND CUDA_NVCC_FLAGS --relocatable-device-code=true)
LIST(APPEND CUDA_NVCC_FLAGS --use_fast_math)

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#
# Lib
#
SET ( 
    libsrc 
    src/arithmetic/polynomial.cu
    src/arithmetic/context.cpp
    src/arithmetic/coprimes.cu
    src/cuda/cudaengine.cu
    src/cuda/sampler.cu
    src/cuda/dgt.cu
    src/tool/log.cu
    src/tool/version.cpp
    src/ckks.cpp
    src/keys.cpp
    src/ckkscontext.cpp
    src/arithmetic/ciphertext.cpp
    src/tool/encoder.cu
)

INCLUDE_DIRECTORIES(
    include
    include/
 )

CUDA_ADD_LIBRARY(aoadgt STATIC ${libsrc})
SET_TARGET_PROPERTIES(aoadgt PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES native)
TARGET_LINK_LIBRARIES(
    aoadgt
    -lntl 
    -lgmp 
    -fopenmp
    -lfftw3
    -lm
    -lssl
    -lcrypto
    ${CUDA_curand_LIBRARY}
)

#
# Testing
#
ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)

#
# Demos
#
# ADD_SUBDIRECTORY(demos)

#
# Install
# 

INSTALL(TARGETS aoadgt
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

INSTALL(FILES include/AOADGT/cuda/cudaengine.h include/AOADGT/cuda/sampler.h include/AOADGT/cuda/dgt.h
    DESTINATION include/AOADGT/cuda
    )

INSTALL(FILES include/AOADGT/arithmetic/polynomial.h include/AOADGT/arithmetic/context.h include/AOADGT/arithmetic/ciphertext.h
    DESTINATION include/AOADGT/arithmetic
    )

INSTALL(FILES include/AOADGT/tool/log.h include/AOADGT/tool/version.h include/AOADGT/tool/encoder.h
    DESTINATION include/AOADGT/tool
    )

INSTALL(FILES include/AOADGT/settings.h include/AOADGT/defines.h include/AOADGT/AOADGTConfig.h include/AOADGT/ckks.h include/AOADGT/ckkscontext.h include/AOADGT/keys.h
    DESTINATION include/AOADGT
    )
