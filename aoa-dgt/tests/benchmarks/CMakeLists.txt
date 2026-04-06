FIND_PACKAGE(AOADGT REQUIRED)

# Benchmark
CUDA_ADD_EXECUTABLE(
    aoadgt_benchmark
    benchmark.cu
)

TARGET_LINK_LIBRARIES(
    aoadgt_benchmark
    aoadgt
    ntl 
    gmp 
    -lssl
    -lcrypto
    ${CUDA_curand_LIBRARY}
)

CUDA_ADD_EXECUTABLE(
    aoadgt_benchmark-residues
    benchmark_residues.cpp
)

TARGET_LINK_LIBRARIES(
    aoadgt_benchmark-residues
    aoadgt
    ntl 
    gmp 
    -lssl
    -lcrypto
    ${CUDA_curand_LIBRARY}
)