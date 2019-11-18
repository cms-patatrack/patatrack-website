---
title: "Working with CUDA in CMSSW"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  heterogeneouscomputing
<!-- choose one among these possible activities: pixeltracks, heterogeneouscomputing, ml -->
---
### Working with CUDA in CMSSW

#### A simple Hello World program

We can write a self-contained program using CUDA and as little of CMSSW as we want; in this example we 
use only the `cudaCheck` function to validate the result of all CUDA calls:
#####`cudaHelloWorld.cu`
```c++
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

__global__
void print(const char * message, size_t length)
{
  //printf("blockIdx.x, threadIdx.x: %d, %d\n", blockIdx.x, threadIdx.x);
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x)
    printf("%c", message[i]);
}

int main(int argc, const char* argv[])
{
  std::string message;
  if (argc == 1) {
    message = "Hello world!";
  } else {
    message = argv[1];
  }

  char * buffer;
  cudaCheck(cudaMalloc(& buffer, message.size()));
  cudaCheck(cudaMemcpy(buffer, message.data(), message.size(), cudaMemcpyDefault));

  print<<<16,1>>>(buffer, message.size());
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;

  print<<<4,4>>>(buffer, message.size());
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;

  print<<<1,16>>>(buffer, message.size());
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;

  cudaCheck(cudaFree(buffer));
}
```

We can build it by hand:
```bash
nvcc -I$CMSSW_BASE/src -m64 -std=c++14 -O3 -g -gencode arch=compute_35,code=sm_35 cudaHelloWorld.cu -o cudaHelloWorld
./cudaHelloWorld
```
or create an entry in a `BuildFile.xml` for it:
#####`BuildFile.xml`
```xml
<bin name="cudaHelloWorld" file="cudaHelloWorld.cu">
  <use name="cuda"/>
</bin>
```
and let SCRAM build it for us:
```bash
scram b
$CMSSW_BASE/test/$SCRAM_ARCH/cudaHelloWorld
```

#### A CMSSW `EDAnalyzer`

We can wrap the same Hello World example in a CMSSW `EDAnalyzer`.
The main change is that we need to split the CUDA code and the framework code, because `nvcc` is not (yet) able to
parse the ROOT include files that the framewrok requires.

We need to put the CUDA kernel and a C++ function (or class) that wraps it in separate .cu / .cuh files:
#####`ScramblerKernel.cuh`
```c++
#ifndef HeterogeneousCore_Examples_test_kernel_cuh
#define HeterogeneousCore_Examples_test_kernel_cuh

#include <cuda_runtime.h>

__global__
void scrambler_kernel(const char * message, size_t length);

void scrambler_wrapper(const char * message, size_t length);

#endif // HeterogeneousCore_Examples_test_kernel_cuh
```

and

#####`ScramblerKernel.cu`
```c++
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

__global__
void scrambler_kernel(const char * message, size_t length)
{
  //printf("blockIdx.x, threadIdx.x: %d, %d\n", blockIdx.x, threadIdx.x);
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x)
    printf("%c", message[i]);
}

void scrambler_wrapper(const char * message, size_t length) {
  scrambler_kernel<<<16,1>>>(message, length);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;

  scrambler_kernel<<<4,4>>>(message, length);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;

  scrambler_kernel<<<1,16>>>(message, length);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;
}
```

We can then call the wrapper from our `EDAnalyzer`:
#####`Scrambler.cc`
```c++
#include <string>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "ScramblerKernel.cuh"

class Scrambler: public edm::stream::EDAnalyzer<> {
public:
  Scrambler(edm::ParameterSet const& config) :
    message_(config.getUntrackedParameter<std::string>("message"))
  {
    cudaCheck(cudaMalloc(& buffer_, message_.size()));
    cudaCheck(cudaMemcpy(buffer_, message_.data(), message_.size(), cudaMemcpyDefault));
  }

  ~Scrambler()
  {
    cudaCheck(cudaFree(buffer_));
  }

  void analyze(edm::Event const&, edm::EventSetup const&)
  {
    scrambler_wrapper(buffer_, message_.size());
  }

private:
  std::string message_;
  char *      buffer_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Scrambler);
```

The last three files (```ScramblerKernel.cuh```, ```ScramblerKernel.cu``` and ```Scrambler.cc```) must be placed inside the ```Package/Subpackage/plugins``` directory. A good way to create the correct subpackage folder skeleton is to follow the steps:

- create your ```Package/``` folder (here called 'UserCode') if you have not done so yet: ```cd $CMSSW_BASE/src/; cmsenv; mkdir UserCode;```

- run the built-in command: ```mkedanlzr <subpackage name>```

Add the following BuildFile.xml fragment inside the ```plugins``` folder as well:
#####`BuildFile.xml`
```xml
<library name="HeterogeneousCoreExamplesScrambler" file="Scrambler.cc ScramblerKernel.cu">
  <use name="cuda"/>
  <use name="FWCore/Framework"/>
  <flags EDM_PLUGIN="1"/>
</library>
```

Finally, add a python configuration file to test the Analyzer:
#####`scrambler.py`
```python
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.scrambler = cms.EDAnalyzer("Scrambler",
    message = cms.untracked.string("Hello world!")
)

process.path = cms.Path( process.scrambler )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 )
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )
)
```

you can store it under ```python```, but its exact location is not important. 
We can then build the new module and run it:
```bash
scram b
cmsRun scrambler.py
```