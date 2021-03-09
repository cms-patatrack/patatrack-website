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

---
# The Intel oneAPI releases

Intel's oneAPI Data Parallel C++ and Libraries are available from the [oneAPI web site]( https://software.intel.com/en-us/oneapi),
and through most package managers.

Currently, oneAPI relies on OpenCL for its CPU, GPU and FPGA backends.
The OpenCL drivers for Intel CPUs and FPGAs are proprietary, and are included
in the oneAPI distribution.
The OpenCL open source "NEO" driver for Intel GPUs is developed on the [Intel Compute Runtime](https://github.com/intel/compute-runtime)
project on GitHub, and can be installed with most package managers.

## RedHat 8.0 and CentOS 8.0

Install the Intel oneAPI base kit and the NEO OpenCL driver with DNF:

```bash
# Intel "NEO" OpenCL drivers
sudo dnf install yum-plugin-copr
sudo dnf copr enable jdanecki/intel-opencl
sudo dnf install intel-opencl

# Intel oneAPI repository
cat << @EOF | sudo tee /etc/yum.repos.d/oneAPI.repo
[oneAPI]
name=Intel(R) oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
@EOF

# Intel oneAPI base kit
sudo dnf install intel-basekit
```

## Ubuntu 18.04 LTS

Install the Intel oneAPI base kit and the NEO OpenCL driver with APT

```bash
# Intel "NEO" OpenCL drivers
sudo add-apt-repository ppa:intel-opencl/intel-opencl
sudo apt install intel-opencl-icd

# Intel oneAPI repository
curl -s https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | sudo apt-key add -
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update

# Intel oneAPI base kit
sudo apt install intel-basekit
```

## Using the Intel Data Parallel C++ compiler

To load the environment to use Intel DPCPP and the CPU and FPGA OpenCL drivers,
it should be enough to run
```bash
source /opt/intel/inteloneapi/setvars.sh
```

---
# The open source Intel LLVM SYCL compiler

## Building the Intel LLVM SYCL compiler

The [sycl branch](https://github.com/intel/llvm/tree/sycl) is under active
development, and is released roughly every two months in the oneAPI betas.

The [devel branch](https://github.com/cms-patatrack/llvm/tree/devel) in the
Patatrack repository tracks the [sycl branch](https://github.com/intel/llvm/tree/sycl)
from Intel, merging some of the open [pull requests](https://github.com/intel/llvm/pulls)
and the latest Patatrack developments.
The differences with respect to the upstream branch can be seen through the
[compare view](https://github.com/intel/llvm/compare/sycl...cms-patatrack:devel)
on GitHub, or from the command line:
```bash
git log --merges --oneline --no-decorate intel/sycl..patatrack/devel
```

Currently they are

   * [#1252](https://github.com/intel/llvm/pull/1252) \[SYCL\] Fixes for multiple backends in the same program
   * [#1288](https://github.com/intel/llvm/pull/1288) \[SYCL\] Run the LIT tests using the selected backend
   * [#1300](https://github.com/intel/llvm/pull/1300) \[SYCL\]\[CUDA\] Fix LIT testing with CUDA devices
   * [#1302](https://github.com/intel/llvm/pull/1302) \[SYCL\]\[CUDA\] Replace assert with CHECK
   * [#1303](https://github.com/intel/llvm/pull/1303) \[SYCL\]\[CUDA\] LIT XFAIL/UNSUPPORTED
   * [#1304](https://github.com/intel/llvm/pull/1304) \[SYCL\]\[CUDA\] LIT exceptions
   * [#1364](https://github.com/intel/llvm/pull/1364) \[SYCL\]\[NFC\] Fix compilation warnings

As for the standard LLVM project, the SYCL compiler can be configured with
`cmake` and built with GNU `make` or `ninja`. The current instructions are at
[GetStartedGuide](https://github.com/cms-patatrack/llvm/blob/devel/sycl/doc/GetStartedGuide.md);
here is a quick summary:
```bash

CUDA_BASE=/usr/local/cuda
SYCL_BASE=$HOME/sycl
INSTALL_PATH=/opt/sycl

cd $SYCL_BASE
git clone https://github.com/cms-patatrack/llvm.git -b devel

mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl;opencl-aot" \
  -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl;opencl-aot;libclc" \
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

Other build options, e.g. `CMAKE_BUILD_TYPE=RelWithDebInfo` or `LLVM_LINK_LLVM_DYLIB=ON`, are also supported.


## Using the Intel LLVM SYCL compiler

To use the compiler directly from the build directory, the minimal setup is
```bash
export PATH=$SYCL_BASE/build/bin:$PATH
export LD_LIBRARY_PATH=$SYCL_BASE/build/lib:$LD_LIBRARY_PATH
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

### Choosing the device type

By default, the SYCL runtime can use any available device: a GPU, a CPU, an
FPGA or other accelerator, or the host device.
The environment variable `SYCL_DEVICE_TYPE` can be used to restrict the runtime
to use a specific device type:
```bash
# force running on a GPU
export SYCL_DEVICE_TYPE=GPU

# force running on a CPU
export SYCL_DEVICE_TYPE=CPU

# force running on an FPGA or other accelerator
export SYCL_DEVICE_TYPE=ACC

# force running on the host device
export SYCL_DEVICE_TYPE=HOST
```

Note that the host device remains available even when selecting a specific
device type.

### Choosing the SYCL backend (OpenCL or CUDA)

The environment variable `SYCL_BE` can be used to instruct the default device
selector to use the OpenCL (default) or CUDA backends:
```bash
# force using the OpenCL backend
export SYCL_BE=PI_OPENCL

# foce using the CUDA backend
export SYCL_BE=PI_CUDA
```

Note that selecting a specific backcend prevents the use of the host device.

### Restricting the available devices

It is possible to restrict the devices available to the SYCL runtime using some
SYCL-specific the environment variables: `SYCL_DEVICE_TYPE` variable (see above),
the `SYCL_DEVICE_ALLOWLIST`, *etc*. See
[EnvironmentVariables.md](https://github.com/cms-patatrack/llvm/blob/devel/sycl/doc/EnvironmentVariables.md)
for more details.

It is also possible to restrict the devices available to the individual backends.
For example, the OpenCL backend may honor the `OCL_ICD_VENDORS` variable (see the
[README.md](https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/master/README.md)
file for the Khronos ICD), and the CUDA backend will honor the `CUDA_VISIBLE_DEVICES`
variable (see the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)).
