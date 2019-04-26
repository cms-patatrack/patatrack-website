# Alpaka and Cupla

## Alpaka
From the Alpaka [README](https://github.com/ComputationalRadiationPhysics/alpaka/blob/develop/README.md):

> The alpaka library is a header-only C++11 abstraction library for accelerator development.
> 
> Its aim is to provide performance portability across accelerators through the abstraction
> of the underlying levels of parallelism.
> 
> It is platform independent and supports the concurrent and cooperative use of multiple devices
> such as the hosts CPU as well as attached accelerators as for instance CUDA GPUs and Xeon Phis
> (currently native execution only). A multitude of accelerator back-end variants using CUDA,
> OpenMP (2.0/4.0), Boost.Fiber, std::thread and also serial execution is provided and can be
> selected depending on the device. Only one implementation of the user kernel is required by
> representing them as function objects with a special interface. There is no need to write
> special CUDA, OpenMP or custom threading code. Accelerator back-ends can be mixed within a
> device queue. The decision which accelerator back-end executes which kernel can be made at
> runtime.

Relevant links:
  - [Alpaka](https://github.com/ComputationalRadiationPhysics/alpaka) on GitHub
  - Alpaka's [documentation](https://github.com/ComputationalRadiationPhysics/alpaka/blob/develop/doc/markdown/user/Introduction.md)

## Cupla

From the Cupla [README]():

> Cupla is a simple user interface for the platform independent parallel kernel acceleration library
> Alpaka. It follows a similar concept as the NVIDIA® CUDA® API by providing a software layer to manage
> accelerator devices. Alpaka is used as backend for Cupla.

Relevant links:
  - [CUPLA](https://github.com/ComputationalRadiationPhysics/cupla) on GitHub
  - CUPLA's [porting guide](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/doc/PortingGuide.md)
  - CUPLA's [tuning guide](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/doc/TuningGuide.md)


# Building with Alpaka and Cupla without CMake

## Set up the environment
```bash
BASE=$PWD
export CUDA_ROOT=/usr/local/cuda-10.0
export ALPAKA_ROOT=$BASE/alpaka
export CUPLA_ROOT=$BASE/cupla

CXX="/usr/bin/g++-7"
CXX_FLAGS="-m64 -std=c++11 -g -O2 -DALPAKA_DEBUG=0 -DCUPLA_STREAM_ASYNC_ENABLED=1 -I$CUDA_ROOT/include -I$ALPAKA_ROOT/include -I$CUPLA_ROOT/include"
HOST_FLAGS="-fPIC -ftemplate-depth-512 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes -Wno-reorder -Wno-sign-compare"

NVCC="$CUDA_ROOT/bin/nvcc"
NVCC_FLAGS="-ccbin $CXX -lineinfo --expt-extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_50,code=sm_50 --use_fast_math --ftz=false --cudart shared"
```

## Download alpaka and cupla
```bash
git clone git@github.com:ComputationalRadiationPhysics/alpaka.git -b 0.3.5 $ALPAKA_ROOT
git clone git@github.com:ComputationalRadiationPhysics/cupla.git  -b 0.1.1 $CUPLA_ROOT
```

## Build cupla ...

### ... for the CUDA backend
```bash
FILES="$CUPLA_ROOT/src/*.cpp $CUPLA_ROOT/src/manager/*.cpp"

mkdir -p $CUPLA_ROOT/build/cuda $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/cuda

for FILE in $FILES; do
  $NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED $CXX_FLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu -c $FILE -o $(basename $FILE).o
done
$NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED $CXX_FLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -shared *.o -o $CUPLA_ROOT/lib/libcupla-cuda.so
```

### ... for the serial backend
```bash
FILES="$CUPLA_ROOT/src/*.cpp $CUPLA_ROOT/src/manager/*.cpp"

mkdir -p $CUPLA_ROOT/build/serial $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/serial

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o
done
$CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-serial.so
```

### ... for the TBB backend
```bash
FILES="$CUPLA_ROOT/src/*.cpp $CUPLA_ROOT/src/manager/*.cpp"

mkdir -p $CUPLA_ROOT/build/tbb $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/tbb

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o
done
$CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -shared *.o -ltbbmalloc -ltbb -lpthread -lrt -o $CUPLA_ROOT/lib/libcupla-tbb.so
```

## Build an example with cupla

### Using CUDA on the gpu
```bash
cd $BASE
$NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED $CXX_FLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -o cuda-vectorAdd -L$CUPLA_ROOT/lib -lcupla-cuda
```

### Using the serial backend on the cpu
```bash
cd $BASE
$CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -o serial-vectorAdd -L$CUPLA_ROOT/lib -lcupla-serial -lpthread
```

### Using the TBB backend on the cpu
```bash
cd $BASE
$CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -o tbb-vectorAdd -L$CUPLA_ROOT/lib -lcupla-tbb -ltbb -lpthread
```
