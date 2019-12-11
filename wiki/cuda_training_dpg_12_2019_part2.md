---
title: "CUDA Training for Tracker DPG - part 2, using CUDA in CMSSW"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
---

## Set up

### Allocation of machines and GPUs

See [part 1](cuda_training_dpg_12_2019.md) for the allocation of a machine and GPU.


### Setting up CMSSW

We are going to use a special release of CMSSW to make use of the utilities
developed by the Patatrack group.  
You can find more information on the [Patatrack development](PatatrackDevelopment.md)
wiki page.

```bash
# on patatrack02 and felk40
export VO_CMS_SW_DIR=/data/cmssw

# on cmg-gpu1080
export VO_CMS_SW_DIR=/data/patatrack/cmssw

# common
export SCRAM_ARCH=slc7_amd64_gcc820

# create a working area
scram list CMSSW_11_0_0_pre13
cmsrel CMSSW_11_0_0_pre13_Patatrack
cd CMSSW_11_0_0_pre13_Patatrack/src

# load the environment
cmsenv

# set up a local git repository
git cms-init -x cms-patatrack
```

You should be in the `from-CMSSW_11_0_0_pre13_Patatrack` branch, and you should
be able to work as you would in a normal CMSSW development area.


### Material for the exercises

```bash
# checkout and build the CMSSW modules used in the tutorial
git cms-merge-topic cms-patatrack:Tutorial_December2019_part2
git cms-addpkg DataFormats/Math Patatrack/Tutorial
scram b -j 4

# generate some random input data
cd Patatrack/Tutorial
cmsRun test/generateCylindricalVectors.py
```


## Simple CUDA modules in CMSSW

### Code organisation

Compilation with CUDA and `nvcc` follows different paths for different files.

A simplified distinction is that

  - **.cc** files are compiled by the host compiler (e.g. `gcc` or `clang`)
  - **.cu** files are compiled by the CUDA compiler driver (`nvcc`) which
      - compiles the host part using the host compiler
      - compiles the device part with an NVIDIA proprietary compiler, after preprocessing with the host compiler
      - links the host and device part, so that the host code can call the kernels on the device

Things become more complicated when we need to split the CUDA code across multiple files, with
[separate compilation and linking](https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/) steps:
![compilation trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/graphics/cuda-compilation-from-cu-to-executable.png)

Things become even more complicated when dealing with shared libraries (device code does not support them) and plugins (_here be dragons_).
SCRAM does its best to support all use cases, but it will need some improvements in this area, as we understand better the constraints.

The option that seems to be working so far is

  - CUDA library calls (e.g. `cudaMalloc()`) can be used anywhere¹
  - CUDA code (e.g. `__global__` and `__device__` functions) should only go in **.cu** files (and in **.h** files included by **.cu** files)
  - **.cu** files should only go in plugins, not in standard shared libraries

We are looking for alternatives, but so far having CUDA kernels in libraries causes wanton chaos and destruction, so don't do it.

___
¹ as long as CUDA is available, which today means: on Intel/AMD and ARMv8 architectures, with CentOS 7 and CentOS 8, with GCC 7.x and 8.x; support for the IBM Power architecture is going to be added, and GCC 9.x should be supported sometimes next year.

### `EDProducer`s and other framewor plugins

A second limitation is that `nvcc` supports c++03, c++11 and c++14 - but not c++17 yet.
Since part of the CMS framework and ROOT are already using features from c++17, we cannot `#include` their headers in **.h** and **.cu** files that are going to be compiled by `nvcc`.

So, we cannot define a framework plugin (e.g. an `EDProducer`) in a ".cu" file; instead we need to split it further, for example in:

  - `plugins/MyEDProducer.cc`: declaration and definition of the plugin;
  - `plugins/MyCUDAStuff.h`: declaration of any CUDA data structures, and plain-C++ wrappers that invoke the kernels;
  - `plugins/MyCUDAStuff.cu`: implementation of device functions, kernels, and of the plain-C++ wrappers that invoke the kernels.

If "MyCUDAStuff" is used only by "MyEDProducer", one can also use the files `plugins/MyEDProducer.h` and `plugins/MyEDProducer.cu` forthe CUDA code.

### Error checking

All CUDA library functions have an error code as their return value.
The safe approach is to wrap *all* calls tu CUDA library function in a wrapper that checks the return value, and throws an exception if there is an error:
```
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

...
  // allocate memory buffers on the GPU
  cudaCheck(cudaMalloc(&gpu_input, sizeof(Input) * input.size()));
  cudaCheck(cudaMalloc(&gpu_output, sizeof(Output) * input.size()));

  // copy the input data to the GPU
  cudaCheck(cudaMemcpy(gpu_input, input.data(), sizeof(Input) * input.size(), cudaMemcpyHostToDevice));
...
```

### Exercise B.1

The `Patatrack/Tutorial` package contains various plugins to

  - generate random 3D vectors in cylindrical coordinates (`GenerateCylindricalVectors`)
  - convert vectors from cylindrical to cartesian coordinates (`ConvertToCartesianVectors`)
  - dump vectors in cylindrical (`PrintCylindricalVectors`) or cartesian (`PrintCartesianVectors`) coordinates
  - compare two collections of vectors in cartesian coordinates (`CompareCartesianVectors`)
  
as well as some configuration files that can be use to run them. For example:
```
cd Patatrack/Tutorial

# generate 1200 events wih 10k random vectors
cmsRun test/generateCylindricalVectors.py

# print the cylindrical vectors in the first event
cmsRun test/printCylindricalVectors.py

# convert the vectors from cylindrical to cartesian in all events
cmsRun test/benchmarkCartesianVectors.py

# convert the vectors from cylindrical to cartesian, and print the cartesian vectors in the first event
cmsRun test/printCartesianVectors.py

# convert the vectors from cylindrical to cartesian using two different modules, and compare their results
cmsRun test/compareCartesianVectors.py
```

As the first exercise:

  - read the `ConvertToCartesianVectors` `EDProducer`
  - using the skeleton privided in `Patatrack/Tutorial/plugins/ConvertToCartesianVectorsCUDA.cc` and `Patatrack/Tutorial/plugins/cudavectors.h`, write a `ConvertToCartesianVectorsCUDA` `EDProducer` that does the same conversion on the GPU
  - use `cmsRun test/compareCartesianVectors.py` to compare the results of the conversion of the CPU and on the GPU; what do you expect ? 
  - vary the precision of the comparison; what do yu see ?


## CUDA framework in CMSSW

For more information, details and examples, read [the documentation](https://github.com/cms-patatrack/cmssw/blob/master/HeterogeneousCore/CUDACore/README.md) prepared by Matti.

## CUDA memory management

### Reusing CUDA memory

### CUDA caching allocators

### Exercise B.2


## Asynchronous execution


### CMSSW ExternalWorker

### CUDA streams

### Exercise B.3


## CPU vs GPU modules

### SwitchProducer mechanism

### Exercise B.4
