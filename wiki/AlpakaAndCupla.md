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

As of December 2019, the latest release of Alpaka is [version 0.4.0](https://github.com/ComputationalRadiationPhysics/alpaka/tree/release-0.4.0).

## Cupla

From the Cupla [README](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/README.md):

> Cupla is a simple user interface for the platform independent parallel kernel acceleration library
> Alpaka. It follows a similar concept as the NVIDIA® CUDA® API by providing a software layer to manage
> accelerator devices. Alpaka is used as backend for Cupla.

Relevant links:
  - [Cupla](https://github.com/ComputationalRadiationPhysics/cupla) on GitHub
  - Cupla's [porting guide](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/doc/PortingGuide.md)
  - Cupla's [tuning guide](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/doc/TuningGuide.md)

As of December 2019, the [dev branch](https://github.com/ComputationalRadiationPhysics/cupla/tree/dev)
of Cupla can be used with Alpaka version 0.4.0.


# Building Alpaka and Cupla without CMake

## Set up the environment
```bash
BASE=$PWD
export CUDA_BASE=/usr/local/cuda
export ALPAKA_BASE=$BASE/alpaka
export CUPLA_BASE=$BASE/cupla

CXX="/usr/bin/g++"
CXXFLAGS="-m64 -std=c++14 -g -O2 -DALPAKA_DEBUG=0 -I$CUDA_BASE/include -I$ALPAKA_BASE/include -I$CUPLA_BASE/include"
HOST_FLAGS="-fopenmp -pthread -fPIC -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes -Wno-reorder -Wno-sign-compare"

NVCC="$CUDA_BASE/bin/nvcc"
NVCC_FLAGS="-ccbin $CXX -w -lineinfo --expt-extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_50,code=sm_50 --cudart shared"
```

## Download Alpaka and Cupla
```bash
git clone git@github.com:ComputationalRadiationPhysics/alpaka.git -b release-0.4.0 $ALPAKA_BASE
git clone git@github.com:ComputationalRadiationPhysics/cupla.git  -b dev           $CUPLA_BASE
```

## Remove the embedded version of Alpaka from Cupla
```bash
cd $CUPLA_BASE
git config core.sparsecheckout true
echo -e '/*\n!/alpaka' > .git/info/sparse-checkout
git read-tree -mu HEAD
cd ..
```

## Build Cupla
```bash
FILES="$CUPLA_BASE/src/*.cpp $CUPLA_BASE/src/manager/*.cpp"
```

### Build Cupla for the serial CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/seq-seq-async
cd $CUPLA_BASE/build/seq-seq-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the serial CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/seq-seq-sync
cd $CUPLA_BASE/build/seq-seq-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the std::threads CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/seq-threads-async
cd $CUPLA_BASE/build/seq-threads-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the std::threads CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/seq-threads-sync
cd $CUPLA_BASE/build/seq-threads-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the OpenMP 2.0 threads CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/seq-omp2-async
cd $CUPLA_BASE/build/seq-omp2-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the OpenMP 2.0 threads CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/seq-omp2-sync
cd $CUPLA_BASE/build/seq-omp2-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the OpenMP 2.0 blocks CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/omp2-seq-async
cd $CUPLA_BASE/build/omp2-seq-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the OpenMP 2.0 blocks CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/omp2-seq-sync
cd $CUPLA_BASE/build/omp2-seq-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the TBB blocks CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/tbb-seq-async
cd $CUPLA_BASE/build/tbb-seq-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the TBB blocks CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/tbb-seq-sync
cd $CUPLA_BASE/build/tbb-seq-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the CUDA GPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/cuda-async
cd $CUPLA_BASE/build/cuda-async

for FILE in $FILES; do
  $NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu -c $FILE -o $(basename $FILE).o &
done
wait
```

### Build Cupla for the CUDA GPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_BASE/build/cuda-sync
cd $CUPLA_BASE/build/cuda-sync

for FILE in $FILES; do
  $NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu -c $FILE -o $(basename $FILE).o &
done
wait
```

### Link all Cupla accelerator backends in a single library
```bash
mkdir -p $CUPLA_BASE/lib
$CXX $CXXFLAGS $HOST_FLAGS $CUPLA_BASE/build/*/*.o -L$CUDA_BASE/lib64 -lcudart -ltbbmalloc -ltbb -shared -o $CUPLA_BASE/lib/libcupla.so
```

# Building an example with Cupla

### Using the serial CPU backend, with synchronous kernel launches
```bash
cd $BASE
$CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS $CUPLA_BASE/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -L$CUPLA_BASE/lib -lcupla -o vectorAdd-seq-seq-sync
LD_LIBRARY_PATH=$CUPLA_BASE/lib ./vectorAdd-seq-seq-sync
```

### Using the TBB blocks CPU backend, with asynchronous kernel launches
```bash
cd $BASE
$CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS $CUPLA_BASE/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -L$CUPLA_BASE/lib -lcupla -ltbbmalloc -ltbb -o vectorAdd-tbb-seq-async
LD_LIBRARY_PATH=$CUPLA_BASE/lib ./vectorAdd-tbb-seq-async
```

### Using the CUDA GPU backend, with asynchronous kernel launches
```bash
cd $BASE
$NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu $CUPLA_BASE/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -L$CUPLA_BASE/lib -lcupla -o vectorAdd-cuda-async
LD_LIBRARY_PATH=$CUPLA_BASE/lib:$CUDA_BASE/lib64 ./vectorAdd-cuda-async
```
