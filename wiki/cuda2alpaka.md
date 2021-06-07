## A first practical example

### Try it yourself

If you want to try to build and run the example you can find the code at: https://github.com/cms-patatrack/pixeltrack-standalone.git

The first step is to change the version of ```gcc``` to 8.3.1.

```bash
source scl_source enable devtoolset-8
```

Then, you need to download the repository.

```bash
# Clone the repository
git clone git@github.com:cms-patatrack/pixeltrack-standalone.git

# Change the current working directory
cd pixeltrack-standalone/
```

The code can be compiled for different backends. In order to compile it for CUDA or Alpaka, the commands are:

```bash
# Build application using cuda-11.3
make -j $(nproc) cuda

# Build application using Alpaka 0.6.0 (currently, Alpaka doesn't work with cuda-11.3)
make -j $(nproc) alpaka CUDA_BASE=/usr/local/cuda-11.2

# Source environment
source env.sh
```

To run the executables, one must call:

```bash
# Process 1000 events in 1 thread using CUDA
./cuda

# Process 1000 events in 1 thread using Alpaka with CUDA backend
./alpaka --cuda

# Process 1000 events in 1 thread using Alpaka with CPU TBB backend
./alpaka --tbb

# Process 1000 events in 1 thread using Alpaka with CPU SERIAL backend
./alpaka --serial
```