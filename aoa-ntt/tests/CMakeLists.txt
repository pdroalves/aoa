FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(newckks REQUIRED)
FIND_PACKAGE(Rapidjson REQUIRED)
FIND_PACKAGE(NTL 11.0 REQUIRED)
FIND_PACKAGE(GMP 6.0)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

#
# TEST
#
ENABLE_TESTING()

CUDA_ADD_EXECUTABLE(
    newckks_test
    tests.cpp
)

TARGET_LINK_LIBRARIES(
    newckks_test
    ${CUDA_curand_LIBRARY}
    ${GTEST_BOTH_LIBRARIES}
    newckks
    -lntl
    -lgmp
)

CUDA_ADD_EXECUTABLE(
    newckks_benchmark
    benchmark.cpp
)

TARGET_LINK_LIBRARIES(
    newckks_benchmark
    ${CUDA_curand_LIBRARY}
    newckks
    -lntl
    -lgmp
)

CUDA_ADD_EXECUTABLE(
    newckks_benchmark_residues
    benchmark_residues.cpp
)

TARGET_LINK_LIBRARIES(
    newckks_benchmark_residues
    ${CUDA_curand_LIBRARY}
    newckks
    -lntl
    -lgmp
)