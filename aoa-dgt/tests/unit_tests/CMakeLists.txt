FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(AOADGT REQUIRED)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

#
# TEST
#
ENABLE_TESTING()

CUDA_ADD_EXECUTABLE(
    aoadgt_arithmetic_test
    test_arithmetic.cu
)

TARGET_LINK_LIBRARIES(
    aoadgt_arithmetic_test
    aoadgt
    ${CUDA_curand_LIBRARY}
    ${GTEST_BOTH_LIBRARIES}
    -lpthread
)

CUDA_ADD_EXECUTABLE(
    aoadgt_ckks_test
    test_ckks.cu
)

TARGET_LINK_LIBRARIES(
    aoadgt_ckks_test
    aoadgt
    ${CUDA_curand_LIBRARY}
    ${GTEST_BOTH_LIBRARIES}
    -lpthread
)