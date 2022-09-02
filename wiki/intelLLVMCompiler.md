# Building and Using Intel's LLVM Compiler (Open Source DPC++)
The open source DPC++ toolchain is Intel's project based on LLVM where the DPC++ features are developed and tested. Here you can find experimental features that have not been released to the commercial DPC++ compiler. However, it comes with  tradeoffs: you have to build the compiler yourself, install dependencies and there are some differences on functionality.

This document provides guidance to build and use the compiler. For more information about SYCl, compilers and backends consult the [SYCL Guide](SYCL.md) or [CUDA2SYCL Porting Guide](cuda2sycl_rules.md).
## Building Compiler with Backends Support
The compiler supports many experimental backends, such as NVPTX and HIP for targeting NVIDIA and AMD devices and they can be added by enabling building flags.

This guide closely follows Intel's [Getting Started with oneAPI DPC++](https://intel.github.io/llvm-docs/GetStartedGuide.html) Guide.

#### Prerequisites
* Windows or Linux OS
* git (tested with v2.31.1)
* cmake v3.14.0+ (tested with v3.20.2)
* python (tested with v3.6.8)
* ninja (tested with v1.8.2)

#### Compiling
Follow the [Getting Started Guide](https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain) "Build DPC++ Toolchain" enabling CUDA or HIP as desired in the configuration step.

## Installing Backend's Low Level Runtimes
For running and compiling SYCL programs it is necessary to have installed the corresponding runtimes. 
* In case of CUDA, having the graphics driver and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is sufficient for compiling in AOT mode SYCL programs. **For doing JIT kernel compilation it is necessary to install a modern CUDA OpenCL implementation and link it.**
* For compiling AOT, JIT and running on Intel CPU's it is necessary to instal the Low Level Runtime. This can be done by following the subsection "Install low level runtime" of the [Getting Started Guide](https://intel.github.io/llvm-docs/GetStartedGuide.html). Perform only the commands necesarry to install OpenCL CPU runtime (ocl_cpu_rt) and TBB library.
    * The links for the runtime binaries and libs can be found in the dependencies file in the Guide.
    * You will need sudo privileges to link the OpenCL runtimes.
* For compiling AOT, JIT and running on Intel GPU's it is necessary to install Intel Graphics Compute Runtime for OpenCL and LevelZero. First install the OpenCL runtime for Intel CPU's and then install the [packages released for the Intel GPU Compiler](https://github.com/intel/compute-runtime/releases) for enabling AOT and JIT compilation or the [Compute Runtime Packages](https://dgpu-docs.intel.com/installation-guides/index.html) for having only JIT compilation.

## Compiling and Running SYCL Applications
DPC++ can either compile all the kernels to target all the backends at runtime (JIT) or during the compilation process (AOT). It is not possible to have a mix of both.

For compiling a SYCL program it is first necessary to have the compiler bin and lib path in environment variables.
```
export PATH=/usr/local/cuda/bin:/data/user/jolivera/sycl_workspace/build/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/data/user/jolivera/sycl_workspace/build/lib:$LD_LIBRARY_PATH
```

Next we can run *clang++* command. To generate an application binary that will compile its kernels at runtime JIT run:
```clang++ -fsycl main.cc [additional-options]```

For compiling all of the program kernels to the target backends at compilation phase, we need to modify the previous command:
```clang++ -fsycl -fsycl-targets=<comma-separated-backends> [-Xsycl-target-backend=<backend> <backend-compiler-options>] [additional-options] main.cc```

- The argument *-fsycl-targets* contains a list of the backends that should be supported and compiled AOT. If this argument is defined, then all desired supported backends must be stated as it disables JIT compilation of kernels.
- The argument *-Xsycl-target-backend* can be used to specify backend-dependant compiler options.
- For compiling in AOT mode for Intel GPU's it is necessary to provide at least one Intel device architecture with the *-device*  argument.

#### Some examples:

* JIT compilation
    ```clang++ -fsycl VectorAdd.cc```
* AOT compilation for Nvidia
    ```clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda VectorAdd.cc```
* AOT compilation for Nvidia with minimun SM 35 supported (default min SM is 50)
    ```clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda "--cuda-gpu-arch=sm_35" VectorAdd.cc```
* AOT compilation for Nvidia and Intel CPU
    ```clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 VectorAdd.cc```
* AOT compilation for only Intel GPU
    ```clang++ -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device xe_hp_sdv" VectorAdd.cc```
* AOT compilation for Nvidia, Intel GPU's and CPU's.
    ```clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen "-device xe_hp_sdv" VectorAdd.cc```

## Things to keep in Mind

#### LLVM compiler differences to DPCPP
Intel LLVM's compiler is based on clang++ so it has natural differences with respect to DPC++ compiler in respect of non standard C++ features:
* Flag *-fsycl* must always be enabled if the source code contains sycl code
* DPC++ custom flags like *-fsycl-enable-function-pointers* are not supported by default (ask Intel maybe)
* It doesn't include implicitly common c++ libraries like *\<optional\>*.
* Some c++ macro attributes are different, for example: *\_\_forceinline* in DPCPP is *\_\_attribute\_\_((always_inline))* in clang++
* Some different syntax for low level asm features, like register variables use only one % instead of %% on declaration.
    * An example can be found in [pixeltrack-standalone/sycl/SYCLCore/prefixScan.h](https://github.com/AuroraPerego/pixeltrack-standalone/blob/Aurora/src/sycl/SYCLCore/prefixScan.h) line 163.
* DPCT and OneAPI extensions headers and libraries are not included by default. They might need to be obtained from a different open source repository.

There are other extra considerations both general and specific for the NVidia backend mentioned in the [Getting Started Guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#known-issues-and-limitations) that should be taken into account.

#### Built Examples
There are 2 built binaries of the LLVM compiler.
* In *patatrack01* VM a built version supports: NVidia and Intel CPU's backend in AOT mode, and Intel GPU and CPU in JIT mode.
* In *patatrack02* VM a built version supports: NVidia backend in AOT mode.

One can also follow this guide in a personal computer. It has been succesfully tested to build a compiler in a Windows 10 computer with WSL2 to support NVidia and Intel's CPU and GPU backends in AOT mode and Intel's CPU and GPU backend in JIT mode.

## References

#### Guides:
* [Getting Started with oneAPI DPC++](https://intel.github.io/llvm-docs/GetStartedGuide.html)
* [SYCL Implementations](SYCL.md)
* [CUDA2SYCL Porting Guide](cuda2sycl_rules.md)

#### Github Repos:
* [Intel DPC++ LLVM](https://github.com/intel/llvm)
* [DPCT Github](https://github.com/oneapi-src/SYCLomatic)
* [Other oneAPI Components](https://github.com/oneapi-src)
* [Intel Graphics Compiler for OpenCL](https://github.com/intel/intel-graphics-compiler)
* [Intel Graphics Compute Runtime for oneAPI LevelZero and OpenCL Driver](https://github.com/intel/compute-runtime)