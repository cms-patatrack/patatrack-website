# Lbraries that extend or work with CUDA

## General purpose libraries

### NVIDIA Toolkit
Installation Guides, Programming Guides, CUDA API References, Documentation and Tools from NVIDIA. 

Documentation: https://docs.nvidia.com/cuda/index.html


### CUDA API Wrappers
Thin C++-flavored wrappers for the CUDA runtime API.

This library of wrappers around the Runtime API is intended to allow us to embrace many of the features of C++ (including some C++11) for using the runtime API - but without reducing expressivity or increasing the level of abstraction (as in, e.g., the Thrust library). Using cuda-api-wrappers, you still have your devices, streams, events and so on - but they will be more convenient to work with in more C++-idiomatic ways.

Documentation: https://codedocs.xyz/eyalroz/cuda-api-wrappers/
Source code: https://github.com/eyalroz/cuda-api-wrappers


## Libraries for Matrix operations
Use cases
  - Riemann Fit:    up to 4x4
  - ECAL multifit:  up to 10x10 (Cholesky) (handlig up to 19x19, bigger for Pahse 2)
  - HCAL MAHI:      up to 10x10 (Cholesky) (handlig up to 19x19)


### NVIDIA cuBLAS / cuSPARSE / cuSOLVER
  - https://developer.nvidia.com/cublas
  - https://docs.nvidia.com/cuda/cublas/index.html
  - https://docs.nvidia.com/cuda/cusparse/index.html
  - https://docs.nvidia.com/cuda/cusolver/index.html


### NVIDIA CUTLASS: Fast Linear Algebra in CUDA C++
CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) at all levels and scales within CUDA. It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement cuBLAS. CUTLASS decomposes these "moving parts" into reusable, modular software components abstracted by C++ template classes.

  - https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/
  - https://github.com/NVIDIA/cutlass/


### Eigen
Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

  - https://eigen.tuxfamily.org/
  - https://bitbucket.org/eigen/eigen/


### MAGMA
The MAGMA project aims to develop a dense linear algebra library similar to LAPACK but for heterogeneous/hybrid architectures, starting with current "Multicore+GPU" systems.

  - http://icl.cs.utk.edu/magma/
  - https://bitbucket.org/icl/magma/


## Other libraries

### NVIDIA CUB
CUB is a flexible library of cooperative threadblock primitives and other utilities for CUDA kernel programming.

  - https://www.microway.com/hpc-tech-tips/introducing-cuda-unbound-cub/
  - https://nvlabs.github.io/cub/
  - https://github.com/nvlabs/cub

### CUDA Data Parallel Primitives Library
CUDPP is a library of data-parallel algorithm primitives such as parallel prefix-sum ("scan"), parallel sort, and parallel reduction. Primitives such as these are important building blocks for a wide variety of data-parallel algorithms, including sorting, stream compaction, and building data structures such as trees and summed-area tables. CUDPP runs on processors that support CUDA.

  - https://cudpp.github.io/
  - https://github.com/cudpp/cudpp


# Libraries for distributed computing

### HPX
Bryce (one of the authors) is now at NVIDIA, so I’m sure there’ll be better support for GPUs in the future

  - http://stellar.cct.lsu.edu/pubs/pgas14.pdf
  - http://stellar.cct.lsu.edu/2018/03/hpx-1-1-0-released/#more-2082

### Legion
There are some folks at NVIDIA who might be able to help

  - http://legion.stanford.edu/overview/index.html
