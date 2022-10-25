PROJECT (newckks LANGUAGES CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

SET(LIB_NAME "newckks")
SET(PROJECT_VERSION_MAJOR "0")
SET(PROJECT_VERSION_MINOR "0")
SET(PROJECT_VERSION_PATCH "4")
SET(PROJECT_VERSION_TWEAK "0")
SET(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${PROJECT_VERSION_TWEAK})")
SET(VERSION ${PROJECT_VERSION})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("\n-- Configuring ${PROJECT_NAME} ${PROJECT_VERSION} (${CMAKE_BUILD_TYPE})...\n")

SET(CMAKE_CXX_FLAGS "-Wall -Wextra")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
SET(CUDA_NVCC_FLAGS_DEBUG "-G")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# configure a header file to pass some of the CMake settings
# to the source code
CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/newckksConfig.h.in"
  "${PROJECT_SOURCE_DIR}/include/newckks/newckksConfig.h"
  )

# add the binary tree to the search path for include files
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")

SET(GCC_CUDA_VERSION "gcc" CACHE STRING "GCC compiler to be used with cuda")

ENABLE_LANGUAGE(CUDA)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(NTL 11.0 REQUIRED)
FIND_PACKAGE(GMP 6.0)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(cxxopts REQUIRED)
FIND_PACKAGE(Doxygen)
FIND_PACKAGE(Rapidjson REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 52)
endif()
LIST(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)
LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread)
LIST(APPEND CUDA_NVCC_FLAGS --relocatable-device-code=true)
LIST(APPEND CUDA_NVCC_FLAGS --use_fast_math)
LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
SET(CUDA_SEPARABLE_COMPILATION ON)

SET(OPTIMIZED_MODUP ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#
# Lib
#
SET ( 
    libsrc 
    src/arithmetic/poly_t.cpp
    src/tool/context.cpp
    src/ckks/ckks.cpp
    src/ckks/ckkscontext.cpp
    src/ckks/cipher_t.cpp
    src/ckks/ckkskeys.cpp
    src/ckks/encoder.cu
    src/cuda/manager.cu
    src/cuda/sampler.cu
    src/cuda/htrans/common.cu
    src/cuda/htrans/ntt.cu
    src/cuda/htrans/coprimes.cu
    src/tool/version.cpp
)

INCLUDE_DIRECTORIES(
    include
    include/
 )

# CKKS mode
CUDA_ADD_LIBRARY(newckks STATIC ${libsrc})
if(DEFINED OPTIMIZED_MODUP)
    MESSAGE("Optimized modup: ON")
    TARGET_COMPILE_DEFINITIONS(newckks PUBLIC OPTMODUP)
else()
    MESSAGE("Optimized modup: OFF")
endif()
TARGET_LINK_LIBRARIES(
    newckks
    ${CUDA_curand_LIBRARY}
    -lntl 
    -lgmp
    -fopenmp
    -lssl
    -lcrypto
)

#
# Testing
#
ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)

#
# Demos
#
#ADD_SUBDIRECTORY(demos)

#
# Install
# 

INSTALL(TARGETS newckks
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

INSTALL(FILES include/newckks/coprimes.h include/newckks/newckksConfig.h include/newckks/defines.h
    DESTINATION include/newckks)

INSTALL(FILES include/newckks/arithmetic/poly_t.h
    DESTINATION include/newckks/arithmetic)

INSTALL(FILES include/newckks/ckks/cipher_t.h include/newckks/ckks/ckks.h include/newckks/ckks/ckkscontext.h include/newckks/ckks/ckkskeys.h include/newckks/ckks/encoder.h
    DESTINATION include/newckks/ckks)

INSTALL(FILES include/newckks/cuda/manager.h include/newckks/cuda/sampler.h
    DESTINATION include/newckks/cuda)

INSTALL(FILES include/newckks/cuda/htrans/common.h include/newckks/cuda/htrans/ntt.h 
    DESTINATION include/newckks/cuda/htrans)

INSTALL(FILES include/newckks/tool/version.h include/newckks/tool/context.h
    DESTINATION include/newckks/tool)
