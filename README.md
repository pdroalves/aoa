# AOA

Research code for **GPU-accelerated CKKS** (approximate homomorphic encryption over the reals). This repository collects two related CMake/C++ projects that implement encrypted polynomial arithmetic and CKKS tooling on CUDA, in the spirit of the SPOG line of work on hierarchical transforms.

## Contents

| Directory | Name | Summary |
|-----------|------|---------|
| [`aoa-dgt/`](aoa-dgt/) | **AOADGT** | CKKS stack built around a **Discrete Galois Transform** (DGT) path. Includes demos, unit tests, and benchmarks. Licensed under **GPL-3.0** (see `aoa-dgt/COPYING`). |
| [`aoa-ntt/`](aoa-ntt/) | **newckks** | CKKS stack using an **NTT** (number-theoretic transform) formulation with CUDA kernels for hierarchical transforms, encoding, and benchmarks. Licensed under the **MIT License** (see `aoa-ntt/LICENSE`). |

Each subdirectory is a standalone CMake project: configure and build from inside `aoa-dgt` or `aoa-ntt` as documented below.

## Requirements

Typical dependencies (exact versions depend on your environment):

- **CMake** ≥ 3.11  
- **CUDA** toolkit and a compatible host compiler  
- **NTL** and **GMP** (see the `cmake/` modules in each project)  
- **OpenMP** (where referenced by the subproject)

## Building

From the repository root, build each component separately:

```bash
cd aoa-dgt
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

cd ../../aoa-ntt
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Adjust `CMAKE_PREFIX_PATH` or package locations if NTL, GMP, or CUDA are installed in non-standard prefixes.

## Related repositories

- **[aoa-logistic-regression](https://github.com/pdroalves/aoa-logistic-regression)** — trains logistic regression on encrypted data using these libraries as the CKKS backend.

## History

Initial import: October 2022.
