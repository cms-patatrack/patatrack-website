# Transition from CUDA to cupla

This is a report on the main steps to convert existing CUDA code to cupla and the main difficulties of the transition. Open problems will be highlighted.

The final goal is to obtain single source code which can be built both for CUDA and for CPU by simply configuring the building process, without modifying the source code. Cupla has been chosen for this task as it defines high level functions, as similar as possible to the CUDA calls, which work on both devices, easing the transition.

## The theory

1. Remove all the CUDA headers and replace them with:

  ```C++
  /* Do NOT include other headers that use CUDA runtime functions or variables
   * (see above) before this include.
   * The reason for this is that cupla renames CUDA host functions and device build in
   * variables by using macros and macro functions.
   * Do NOT include other specific includes such as `<cuda.h>` (driver functions,
   * etc.).
   */
  #include <cuda_to_cupla.hpp>
  ```

2. Transform the CUDA kernels to cupla functors. This is done by transforming the ``__global__`` functions to structures with a templated const ``operator()``, which must have the ``ALPAKA_FN_ACC`` prefix and a first templated argument called ``acc``. This is used by cupla to pass all the accelerator specific information.

 **CUDA kernel**
  ```C++
  template< int blockSize >
  __global__ void fooKernel( int * ptr, float value )
  {
      // ...
  }
  ```

  **cupla kernel**
  ```C++
  template< int blockSize >
  struct fooKernel
  {
      template< typename T_Acc >
      ALPAKA_FN_ACC
      void operator()( T_Acc const & acc, int * const ptr, float const value) const
      {
          // ...
      }
  };
  ```

3. Transform the host side kernel calls by using the ``CUPLA_KERNEL`` macro.

  **CUDA host side kernel call**
  ```C++
  // ...
  dim3 gridSize(42,1,1);
  dim3 blockSize(256,1,1);
  // extern shared memory and stream is optional
  fooKernel< 16 ><<< gridSize, blockSize, 0, 0 >>>( ptr, 23 );
  ```

  **cupla host side kernel call**
  ```C++
  // ...
  dim3 gridSize(42,1,1);
  dim3 blockSize(256,1,1);
  // extern shared memory and stream is optional
  CUPLA_KERNEL(fooKernel< 16 >)( gridSize, blockSize, 0, 0 )( ptr, 23 );
  ```

4. Transform ``__device__`` functions to ``ALPAKA_FN_ACC`` and add the ``acc`` templated parameter. This is needed only when `alpaka` functions like `blockIdx` or `atomicMax`, ... are used.


**CUDA**
  ```C++
  template< typename T_Elem >
  __device__ int deviceFunction( T_Elem x )
  {
      // ...
  }
  ```

**cupla**
  ```C++
  template< typename T_Acc, typename T_Elem >
  ALPAKA_FN_ACC int deviceFunction( T_Acc const & acc, T_Elem x )
  {
      // ...
  }
  ```

  Lastly, remember to add the ``acc`` parameter to the function call. 

The full guide can be found on the [cupla repository](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/doc/PortingGuide.md).

## A first practical example

As a first test case, the `RecoPixelVertexing/PixelTrackFitting/test/testEigenGPUNoFit.cc` of cmssw was ported. This example uses eigen as en external library but is overall independent from the rest of the codebase.

Here you will find some of the problems found and their solution.

### Build configuration

The working build configuration is the following:

```xml
<bin file="testEigenGPUNoFit.cu" name="testEigenGPUNoFit_t">
  <use name="eigen"/>
  <use name="cuda"/>
  <use name="cuda-api-wrappers"/>
  <use name="HeterogeneousCore/CUDAUtilities"/>
  <use name="boost"/>
  <flags CXXFLAGS="-g -I/<path_to_cupla>/include/ -I<path_to_alpaka>/include -D FOR_CUDA"/>
</bin>

<bin file="testEigenGPUNoFit.cc" name="testEigenGPUNoFit_c">
  <use name="eigen"/>
  <use name="boost"/>
  <flags CXXFLAGS="-g -I/<path_to_cupla>/include/ -I<path_to_alpaka>/include"/>
</bin>
```

Cupla and Alpaka are header only libraries and do not need to be builded. The `dev` branches of `cms-patatrack/cupla` has been used as it includes some PR to make it header only. For this example they are included using directly the `-I` flag, but eventually they will have to be integrated like boost or eigen. 

Since `scram` deduces the compiler to use (g++ o nvcc) from the file extensions, both the .cc and .cu file must be present. Since we don't want to keep a copy of the same file, a symbolic link has been used. Ultimately, the build tools should be updated to handle this case without the use of a symbolic link.

### Makefile

Alternatively, one could write a Makefile to build this example without the cmssw environment. Boost is required.

```Makefile
.PHONY: all clean

TARGETS := test-cuda test-host

CXX         := /usr/bin/g++-8
CXXFLAGS    := -std=c++14 -O2 -g -pthread

# Eigen configuration
EIGEN_BASE := <path_to_eigen>
EIGEN_FLAGS := -I$(EIGEN_BASE)/include/eigen3 -D EIGEN_DONT_PARALLELIZE

# CUDA configuration
CUDA_BASE   := /usr/local/cuda-10.1
CUDA_FLAGS  := -x cu -std=c++14 -O2 -g -w --expt-relaxed-constexpr --compiler-options "-pthread"
CUDA_CXX    := $(CUDA_BASE)/bin/nvcc

# Alpaka/Cupla configuration
ALPAKA_BASE := <path_to_alpaka>
CUPLA_BASE  := <path_to_cupla>
CUPLA_FLAGS := -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include 

all: $(TARGETS)

clean:
  rm -f $(TARGETS)

# build Cuda Cupla code
test-cuda: testEigenGPUNoFit.cc
  $(CUDA_CXX) $(CUDA_FLAGS) $(CUPLA_FLAGS) $(EIGEN_FLAGS) $^ -o $@

# build Cupla Host code
test-host: testEigenGPUNoFit.cc
$(CXX) $(CXXFLAGS) $(CUPLA_FLAGS) $(EIGEN_FLAGS) $^ -o $@
```

### Headers

Cupla porting guide suggests to remove all the headers related to CUDA and replace them with `<cuda_to_cupla.hpp>`. In practice, we found that we needed to include also device specific headers (eg: `<cupla/standalone/CpuSerial.hpp>` for CPU and `<cupla/standalone/GpuCudaRt.hpp>` for CUDA). Since we want a single source file that can be built for both devices, we created a `cms_cupla.h` header with includes one of the two depending on the target device.

**cms_cupla.h**
```c++
#ifdef FOR_CUDA
#include <cupla/standalone/GpuCudaRt.hpp>
#else
#include <cupla/standalone/CpuSerial.hpp>
#endif
#include <cuda_to_cupla.hpp>
```

By including this file instead of `<cuda_to_cupla.hpp>` we managed to use a single header file for all the target devices. This requires to declare the `FOR_CUDA` macro in the build configuration, as can be seen in the above examples. 

When using external libraries which uses CUDA internally, as eigen, **the order of the includes is important**. We realized that by including Eigen after cupla, the macro definitions of cupla completely broke eigen, thus it is important to include it earlier.

This is something to keep in mind when considering moving a large codebase with many dependencies as all of those may have to be patched to be *cupla compliant*.

### CUDA code porting

Porting CUDA code was straightforward, it is sufficient to follow the porting guide. One thing to notice is that many CUDA functions can be left untouched, as they are defined by cupla with the same signature. This includes `cudaMalloc`, `cudaMemcpy` and `cudaDeviceSynchronize`.

Perl regular expressions can be used to speedup the code conversion, in particular for the most common cases, but use with caution and always check the result.

**kernels**
```bash
perl -0777 -i -pe 's/\s+__global__\s+void\s+(\w+)\s*\((.*?)\)/\nstruct $1\n{\n  template< typename T
_Acc >\n  ALPAKA_FN_ACC\n  void operator()( T_Acc const & acc, $2 ) const/igs' source_file.cc
```
After this conversion, you have to manually close each new struct with `};`

**host device functions**
```bash
perl -0777 -i -pe 's/(.*?)__host__\s+/$1/igs' source_file.cc

perl -0777 -i -pe 's/(.*?)__device__/$1ALPAKA_FN_ACC/igs' source_file.cc
```

A problem was found when dealing with `printIt` function as it is defined as a `__host__ __device__` function inside `test_common.h` which is used by many tests. This means that porting it to cupla would have broken all the other tests. For this example these calls (printf while in debug mode) have been commented. This gives the idea that it will be difficult to port just portions of a more complex codebase to cupla, given all the internal dependencies.

### Try it yourself

If you want to try to build and run this example you can find the code at: https://github.com/darcato/cmssw/tree/my_development_branch

You first have to download cupla and alpaka:

```bash
cd my_ws
git clone -b dev https://github.com/cms-patatrack/cupla.git
git clone -b develop https://github.com/ComputationalRadiationPhysics/alpaka.git
```

Since these are headers only, just update the paths on the `BuildFile.xml` located inside `src/RecoPixelVertexing/PixelTrackFitting/test`. Then you have to setup the cmssw environment and run `scram b` from that folder. Remember to use a newer version of gcc to avoid internal compiler errors:

```bash
export SCRAM_ARCH=slc7_amd64_gcc820
```

### Example takeaway

This first example proved the feasibility of porting a piece of code from CUDA to cupla, but still highlighted multiple problems to be solved. The code is built and runs correctly but many warning are still present, for example:

```
/data/user/dmarcato/alpaka/include/alpaka/mem/view/ViewPlainPtr.hpp(91): warning: __host__ annotation is ignored on a function("~ViewPlainPtr") that is explicitly defaulted on its first declaration

/data/cmssw/slc7_amd64_gcc820/external/eigen/e4c107b451c52c9ab2d7b7fa4194ee35332916ec-pafccj/include/eigen3/Eigen/src/Core/products/GeneralBlockPanelKernel.h(1917): warning: calling a __host__ function from a __host__ __device__ function is not allowed
```

## Porting production code from cmssw

After the first example described above, the porting of a plugin from `RecoLocalTracker/SiPixelClusterizer` was attempted. 

The difference with the previous one is that now the code has much more dependencies spread all over the cmssw codebase. This means that to build just one file, many different parts of the project had to be updated. In practice the first step of the conversion process is to identify all the included files and apply the porting procedures as explained above. Here the use of a script helps a lot.

### Main problems encountered

1. Cupla does not define `__syncthreads_or` and `__syncthreads_and`, even though they are defined by alpaka. This was solved by adding these lines to `/cupla/cudaToCupla/driverTypes.hpp`:

```c++
#define __syncthreads_or(...) ::alpaka::block::sync::syncBlockThreadsPredicate<::alpaka::block::sync::op::LogicalOr>(acc, __VA_ARGS__)
#define __syncthreads_and(...) ::alpaka::block::sync::syncBlockThreadsPredicate<::alpaka::block::sync::op::LogicalAnd>(acc, __VA_ARGS__)
```

2. `cudaCompact.h` tries to do the same job as cupla, that is exposing functions which can work both on the host and with CUDA. This has to be avoided when using cupla, so the code is placed inside an `#ifdef` which removes it when using cupla:

```
#if !defined __CUDACC__  &&  !defined CUPLA_KERNEL
```

3. `cudaHostAlloc()` is not defined by cupla. It can be replaced with `cudaMallocHost()` but special attention has to be put on the arguments.
4. `cuda::throw_if_error()` is not defined as it is a function from `<cuda/api_wrappers.h>`. All CUDA code from third party libraries may need to be patched to support cupla.

This example shows that it is difficult to port just a small portion of code and that third party libraries complicate the transition. A first step to ease the transition is to write the code in pure CUDA, removing all the parts that would not be necessary anymore when using cupla, and then proceeding to the porting.


Davide Marcato 07/2019