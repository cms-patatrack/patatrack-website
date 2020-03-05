# SYCL Overview

From the Khronos Group's SYCL [web page](https://www.khronos.org/sycl/):

> SYCL (pronounced 'sickle') is a royalty-free, cross-platform abstraction layer
> that builds on the underlying concepts, portability and efficiency of OpenCL 
> that enables code for heterogeneous processors to be written in a "single-source"
> style using completely standard C++. SYCL single-source programming enables
> the host and kernel code for an application to be contained in the same source
> file, in a type-safe way and with the simplicity of a cross-platform asynchronous
> task graph.

Broasly speaking, SYCL is to OpenCL and SPIR/SPIR-V what the CUDA runtime API is
to to the CUDA driver API and PTX.

The current version of SYCL is the [SYCL 1.2.1 specification, revision 6](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf),
released on November 14, 2019, and is based on OpenCL 1.2.

A [provisional SYCL 2.2 specification](https://www.khronos.org/registry/SYCL/specs/incomplete_deprecated_provisional_sycl-2.2.pdf)
was published in February 2016. That specification was an incomplete work in
progress, and should be considered deprecated.

![/sycl-targets](https://raw.githubusercontent.com/illuhad/hipSYCL/master/doc/img/sycl-targets.png)
*different SYCL toolchains and the backends they support, from the
[hipSYCL README](https://github.com/illuhad/hipSYCL/blob/master/README.md)*

Different tool chains implement the SYCL specification for diverse targets,
extending it beyond the original scope of OpenCL 1.2:

   * [ComputeCpp](https://developer.codeplay.com/products/computecpp/ce/home/):
     SYCL implementation by Codeplay, targeting OpenCL devices with SPIR support
     and (experimentally) NVIDIA GPUs via a PTX backend.
   * [triSYCL](https://github.com/triSYCL/triSYCL): an open-source implementation
     to experiment with SYCL, targeting OpenCL devices with SPIR+ support, such as
     [PoCL](http://portablecl.org/) and Xilinx FPGAs
   * [hipSYCL](https://github.com/illuhad/hipSYCL): a project to develop a SYCL 
     1.2.1 implementation that builds upon NVIDIA CUDA and AMD HIP.
   * [Intel Data Parallel C++](https://software.intel.com/en-us/oneapi/dpc-compiler):
     a SYCL 1.2.1 implementation with extensions to support many CUDA-inspired
     features, developed as an [open source project](https://github.com/intel/llvm).
     It uses a plugin mechanism to target different backends, currently OpenCL with
     SPIR-V support and CUDA with PTX.


## Building the Intel LLVM SYCL compiler

The [sycl branch](https://github.com/intel/llvm/tree/sycl) is under active
development, and is released roughly every two months in the oneAPI betas.
These instruction are current as of March 5th, 2020, but will likely be partially
obsolete very soon.

Building the CUDA backend requires a recent installation of the CUDA toolkit;
it has been tested with CUDA 10.1.243 and 10.2.89 on CentOS Linux 8.1.1911.

Merging the open pull requests locally may give raise to some conflicts, but so
far they were only about differences in indentation and alignment, and should
be easy to fix.

```bash
CUDA_BASE="/usr/local/cuda"
SYCL_BASE="/data/user/fwyzard/sycl"
INSTALL_PATH="/opt/sycl/latest"
BUILD_TYPE="RelWithDebInfo"                         # valid values: Release, RelWithDebInfo, Debug

git clone git@github.com:intel/llvm.git -b sycl $SYCL_BASE/llvm
cd $SYCL_BASE/llvm

git fetch origin pull/1181/head
git merge FETCH_HEAD -m '[SYCL][CUDA] Implements program compile and link (#1181)'
git fetch origin pull/1228/head
git merge FETCH_HEAD -m '[SYCL][Test] Fix SYCL library location path for LIT tests (#1228)'
git fetch origin pull/1241/head
git merge FETCH_HEAD -m '[SYCL][CUDA] Implement part of USM (#1241)'
git fetch origin pull/1252/head
git merge FETCH_HEAD -m '[SYCL] Fixes for multiple backends in the same program (#1252)'

mkdir -p $SYCL_BASE/build
cd $SYCL_BASE/build

cmake \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl;opencl-aot" \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;compiler-rt;lld;openmp;llvm-spirv;sycl;opencl-aot;libclc" \
  -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_BASE/llvm/sycl \
  -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_BASE/llvm/llvm-spirv \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_PIC=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLIBCLC_TARGETS_TO_BUILD="nvptx64--;nvptx64--nvidiacl" \
  -DSYCL_BUILD_PI_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_BASE \
  -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
  $SYCL_BASE/llvm/llvm

make -j`nproc` sycl-toolchain opencl-aot
```

### Notes

The cmake option `-DLLVM_LIBDIR_SUFFIX=64` is not fully tested, even if it
should be usable for the build.

The cmake option `-DBUILD_SHARED_LIBS=ON` is not supported by (at least) the
CUDA backend.

It should be possible to install the toolchain with `make deploy-sycl-toolchain
deploy-opencl-aot`, but
   - the `LLVM_LIBDIR_SUFFIX` option is not properly taken into account;
   - the CUDA backend is not installed;
so the best option for the moment is to use the `build` directory directly, or 
to symlink it to the installation directory:
```bash
ln -s $SYCL_BASE/build $INSTALL_PATH
```

## Using the Intel LLVM SYCL compiler

To use the compiler directly from the build directory, the minimal setup is
```bash
export PATH=$SYCL_BASE/build/bin:$PATH
export LD_LIBRARY_PATH=$SYCL_BASE/build/lib:$LD_LIBRARY_PATH
```

If the build has also been installed (or symlinked), replace `$SYCL_BASE/build`
with `$INSTALL_PATH`:
```bash
export PATH=$INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$LD_LIBRARY_PATH
```


To build an application for both OpenCL and CUDA backends, use
```
clang++ -O2 -g -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice,spir64-unknown-linux-sycldevice source.cc -o a.out
```

To specify a CUDA GPU architecture, e.g. SM 3.5 (the default is SM 3.0), add
```
... -Xsycl-target-backend=nvptx64-nvidia-cuda-sycldevice '--cuda-gpu-arch=sm_35'`
```

To silence the warnings about an unknouwn CUDA version (e.g. 10.2), add 
```
... -Wno-unknown-cuda-version`
```

### Ahead-of-time compilation

The default behaviour is to compile the OpenCL kernels from the SPIR-V
intermediate representation to the actual binaries *just in time* (JIT), the
first time they are loaded and run.
It is possible to request this compilation *ahead of time* (AOT) with dedicated
SYCL targets.

To compile the OpenCL binary for Intel 64-bit CPUs, add the target
```
... -fsycl-targets=...,spir64_x86_64-unknown-linux-sycldevice
```

To compile the OpenCL binary for specific Intel GPUs, add the target and options
```
... -fsycl-targets=...,spir64_gen-unknown-linux-sycldevice -Xsycl-target-backend=spir64_gen-unknown-linux-sycldevice '-device gen9'`
```

To compile the OpenCL binary for Intel FPGAs (which requires to have the Intel
`aoc` tool available in the `$PATH`), add the target and options
```
... -fsycl-targets=...,spir64_fpga-unknown-linux-sycldevice -lOpenCL -MMD
```

### The SYCL default device

*This section needs to be added.*

TL;DR:

   * `SYCL_BE=PI_OPENCL`
   * `SYCL_BE=PI_CUDA`

### Restricting the available devices

*This section needs to be added.*

TL;DR:

   * `OCL_ICD_VENDORS=...`
   * `CUDA_VISIBLE_DEVICES=...`
   * `SYCL_DEVICE_TYPE=...`
   * `SYCL_DEVICE_ALLOWLIST=...`
