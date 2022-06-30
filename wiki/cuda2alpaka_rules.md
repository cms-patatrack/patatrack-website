# CUDA to Alpaka rules for CMSSW

## CUDADataFormats to AlpakaDataFormats
### General rules
- Include the following:
```c++=
#include "AlpakaCore/alpakaConfig.h"
```

- Wrap the class under the namespace ```ALPAKA_ACCELERATOR_NAMESPACE```
```c++=+
namespace ALPAKA_ACCELERATOR_NAMESPACE {
    class MyClass {
     ...
    }
}
```

## Memory allocations
### Buffer types declaration
- Objects declared as pointers in CUDA must be declared as **alpaka buffers**
    - host: `cms::alpakatools::host_buffer<classObject>`
    - device: `cms::alpakatools::device_buffer<Device, classObject>`
### Host buffers
- **non-cached** and **non-pinned**, equivalent to `malloc()`
    
    - `cms::alpakatools::make_host_buffer<T>()`: scalar buffer
    - `cms::alpakatools::make_host_buffer<T[extent]>()`: 1-dimensional buffer
    - `cms::alpakatools::make_host_buffer<T[]>(extent)`: 1-dimensional buffer
- **potentially cached** and **pinned**
The memory is pinned according to the device associated to the **queue**. 
    - `cms::alpakatools::make_host_buffer<T>(queue)`: scalar buffer
    - `cms::alpakatools::make_host_buffer<T[extent]>(queue)`: 1-dimensional buffer
    - `cms::alpakatools::make_host_buffer<T[]>(queue, extent)`: 1-dimensional buffer  
### Host views
#### If you want to use already allocated memory, you can wrap it within a view!
- `cms::alpakatools::make_host_view<T>(data)`: view of a scalar buffer
- `cms::alpakatools::make_host_view<T[extent]>(data)`: view of a 1-dimensional buffer
- `cms::alpakatools::make_host_view<T[]>(data, extent)`: view of a 1-dimensional buffer 
### Device buffers
- always **pinned**, **potentially cached**
    - `cms::alpakatools::make_device_buffer<T>(queue)` scalar buffer
    - `cms::alpakatools::make_device_buffer<T[extent]>(queue)`: 1-dimensional buffer
    - `cms::alpakatools::make_device_buffer<T[]>(queue, extent)`: 1-dimensional buffer

## NOTE
Currently, **the Alpaka buffers have not a default constructor.**.
If you need a buffer object as a class member, you need to wrap it around `std::optional`. 
#### std::optional
```c++=
class MyClass {

    MyClass() = default;
        
    void initialize(Queue& queue){
        myBuf = cms::alpakatools::make_host_buffer<float[]>(stream, extent);
    }
        
    private:
        std::optional<cms::alpakatools::host_buffer<float[]>> myBuf;
}
```
**OR** delete the default constructor and define a new constructor and initialize the buffer
#### delete default constructor

```c++=
class MyClass {

    MyClass() = delete;
    explicit MyClass(Queue& queue) 
        : myBuf{cms::alpakatools::make_host_buffer<float[]>(stream, extent)}
            {};
        
    private:
        cms::alpakatools::host_buffer<float[]> myBuf;
}
```

### Device views
#### If you want to use already allocated memory, you can wrap it within a view!
- `cms::alpakatools::make_device_view<T>(data, device)`: view of a scalar buffer
- `cms::alpakatools::make_device_view<T[extent]>(data, device)`: view of a 1-dimensional buffer
- `cms::alpakatools::make_device_view<T[]>(data, device, extent)`: view of a 1-dimensional buffer 

The `device` is obtained from the queue: `alpaka::getDev(queue)`

:::warning
Note: The view does not own the underlying memory, make sure that the view does not outlive its underlying memory!
:::

## Memory copy / set
- `alpaka::memcpy(queue, dest_buffer_or_view, source_buffer_or_view)`
- `alpaka::memset(queue, buffer_or_view, value)` : set the whole buffer/view to `value`

:::info
The synchronization behavior in alpaka is defined by the **queue**, which has a *Blocking* or *NonBlocking* property (specified when the object is created).
:::


## Usage example - Buffers 
#### CUDA
```c++=
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

cms::cuda::device::unique_ptr<Object> ptr_d;
ptr_d{cms::cuda::make_device_unique<Object>(/*extent*/, stream)};

cms::cuda::device::unique_ptr<Object> ptr_h;
ptr_h{cms::cuda::make_host_buffer<Object>(/*extent*/, stream)};
```

#### ALPAKA
```c++=
#include <alpaka/alpaka.hpp>
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"

//initialize empty host buffer
cms::alpakatools::host_buffer<Object> myBuf{
                cms::alpakatools::make_host_buffer<Object>(queue, /*extent*/)
                };

//initialize empty device buffer
cms::alpakatools::device_buffer<Device, Object> myDeviceBuf{
                cms::alpakatools::make_device_buffer<Object>(Queue, /*extent*/)
                };
```

## Usage example - Views 
```c++=
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"

/* myDeviceBuf already allocated before */

int * x = myDeviceBuf.data();
auto x_view_device = cms::alpakatools::make_device_view<int>(device, x, extent);
// device obtained using alpaka::getDev(queue);
auto x_buf_host = cms::alpakatools::make_host_buffer<int>(queue, extent);

alpaka::memcpy(queue, x_buf_host, x_view_device);
```
**Note**:
One can request the device from the queue using ```alpaka::getDev(queue)```!

The same logic is applied using the`View` on the host and making the copy on the device. The copy can also happen between `View`s. 

## Heterogeneous DataFormats
### CUDA
Heterogeneous unique pointer interface:`HeterogeneousSOA`

```c++=
#include "CUDADataFormats/HeterogeneousSoA.h"
class SoA {
...
}
using HeterogeneousObject = HeterogeneousSoA<SoA>;
```

GPU Object: ```HeterogeneousObject```
CPU Object: ```cms::cuda::host::unique_ptr<SoA>```

### Alpaka
Different buffers for Host and Device
#### Object definition
In **AlpakaDataFormats**

*AlpakaDataFormats/SoAObject.h*
```c++=
class SoAObject{
    ...
}
```

#### Host specialization
In **AlpakaDataFormats** 

*AlpakaDataFormats/SoAObjectHost.h*

```c++=
#ifndef AlpakaDataFormats_SoAObject_h
#define AlpakaDataFormats_SoAObject_h

#include AlpakaDataFormats/SoAObject.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"

using SoAObjectHost = cms::alpakatools::host_buffer<SoAObject>;

#endif // AlpakaDataFormats_SoAObject_h
```

#### Device specialization
:::warning
:exclamation: In **AlpakaDataFormats/alpaka/** :exclamation: 
:::
*AlpakaDataFormats/alpaka/SoAObjectDevice.h*
```c++=
#define AlpakaDataFormats_SoAObjectDevice_h
#ifndef AlpakaDataFormats_SoAObjectDevice_h

#include AlpakaDataFormats/SoAObject.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    using SoAObjectDevice = cms::alpakatools::device_buffer<SoAObject>;
    
}
#endif // AlpakaDataFormats_SoAObjectDevice_h
```

<!-- - Heterogeneous DataFormats
    - Andrea Alpaka-Collections ??? -->

## Product, Context, ScopedContext


| CUDA                               | ALPAKA                                          |
|:---------------------------------- |:----------------------------------------------- |
| `cms::cuda::ContextState`          | `cms::alpakatools::ContextState<Queue>`         |
| `cms::cuda::Product<Object>`       | `cms::alpakatools::Product<Queue, Object>`      |
| `cms::cuda::ScopedContextAcquire`  | `cms::alpakatools::ScopedContextAcquire<Queue>` |
| `cms::cuda::ScopedContextProduce` | `cms::alpakatools::ScopedContextProduce<Queue>`                                           |

## ESProduct and ESProducers
As a reminder, the job of an ESProducer is to add data to one or more EventSetup Records. 

### ESProduct
Same logic applied in CUDA, conditions can be transferred to the device with the following pattern
- Define a `class`/`struct` for the data to be transferred
- Define a ESProduct wrapper that holds the data
- The wrapper should have a function that transfer the data to the device, asynchronously

### Example 

Define the `class`/`struct` in `CondFormats/`

`CondFormats/ESProductExampleAlpaka.h`
```c++=
#ifndef CondFormats_ESProductExampleAlpaka_H
#define CondFormats_ESProductExampleAlpaka_H

struct PointXYZ {
    float x;
    float y;
    float z;
};

struct ESProductExampleAlpaka {
    PointXYZ* someData;
    unsigned int size;
};

#endif
```
Define its wrapper in `CondFormats/alpaka/`. The corresponding ESProducer should produce objects of this type.

`CondFormats/alpaka/ESProductExampleAlpakaWrapper.h`
```c++=
#ifndef CondFormats_alpaka_ESProductExampleAlpakaWrapper_h
#define CondFormats_alpaka_ESProductExampleAlpakaWrapper_h

#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "CondFormats/ESProductExampleAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
    class ESProductExampleAlpakaWrapper {
    public:
      // Constructor initialize the internal buffer on the CPU
      ESProductExampleCUDAWrapper(PointXYZ const& someDataInit, unsigned int size) :
           someData_{cms::alpakatools::make_host_buffer<PointXYZ>()},
           size_{size} 
           {
               *someData_ = someDataInit;
               
           }

      // Deallocates all pinned host memory
      ~ESProductExampleAlpakaWrapper() = default;

      // Function to return the actual payload on the memory of the current device
      ESProductExampleAlpaka const *getGPUProductAsync(Queue& queue) const {
          const auto& data = gpuData_.dataForDeviceAsync(queue, [this](Queue& queue) {
              //initialize GPUData, no default constructor
              GPUData gpuData(queue, size_);
              //memcpy from CPU buffer to GPUData internal buffers on the device
              alpaka::memcpy(queue, gpuData.someDataDevice, someData_);
              //fill internal buffer of struct on the CPU
              gpuData.esProductHost->someData = gpuData.someDataDevice.data();              
              //final copy of struct from host to device
              alpaka::memcpy(queue, gpuData.esProductDevice, gpuData.esProductHost);
              return gpuData;
          });
        // return the class/struct on current device
        return data.esProductDevice.data();
      };

    private:
      // Holds the data in pinned CPU memory
      cms::alpakatools::host_buffer<PointXYZ> someData_;      
      unsigned int size_;

      // Helper struct to hold all information that has to be allocated and
      // deallocated per device
      struct GPUData {
          public:
            GPUData () = delete; // alpaka buffers have not default constructor
            GPUData(Queue& queue, unsigned int size):
                esProductHost{cms::alpakatools::make_host_buffer<ESProductExampleAlpaka>(queue)},
                esProductDevice(cms::alpakatools::make_device_buffer<ESProductExampleAlpaka>(queue)),
                someDataDevice{cms::alpakatools::make_device_buffer<PointXYZ>(queue)}
                {};
            // Destructor should free all member pointers, automatic in alpaka
            ~GPUData() = default; 
          public:
            // internal buffers are on device, struct itself is on CPU
            cms::alpakatools::host_buffer<ESProductExampleAlpaka> esProductHost;
            // struct on the device
            cms::alpakatools::device_buffer<Device, ESProductExampleAlpaka> esProductDevice;
            //internal buffers
            cms::alpakatools::device_buffer<Device, PointXYZ> someDataDevice;
      };
      // Helper that takes care of complexity of transferring the data to
      // multiple devices
      cms::alpakatools::ESProduct<Queue, GPUData> gpuData_;
}


#endif
```
### ESProducer
If the object to be produced uses Alpaka: Add the file in `plugin-MyPlugin/alpaka/` Wrap the class around `ALPAKA_ACCELERATOR_NAMESPACE`, define the Framework Module with: 
```DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(MyESProducer)```.

```c++=
#include "AlpakaCore/alpakaConfig.h"
#include "CondFormats/MyCondObject.h"
#include "CondFormats/alpaka/MyAlpakaCondObject.h"
#include "Framework/ESPluginFactory.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
    class MyESProducer : public edm::ESProducer {
        // MyESProducer Constructor
        void produce(/*args*/);
    };
    ...
} // namespace ALPKA_ACCELERATOR_NAMESPACE
DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(MyESProducer);
```


## EDProducer and EDProducerExternalWork

### General rules
Wrap the class around `ALPAKA_ACCELERATOR_NAMESPACE`
```c++=
#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaConfig.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE{
    class MyProducer : public edm::EDProducerExternalWork // or EDProducer 
    {
        ...
    };
} // namespace ALPAKA_ACCELERATOR_NAMESPACE
DEFINE_FWK_ALPAKA_MODULE(MyProducer);
```



## Alpaka kernels
Alpaka kernels are defined as C++ functors whose call is templated on the type of accelerator:
```c++
struct myAlpakaKernel {
template<typename TAcc>
ALPAKA_FN_ACC void operator()(const TAcc &acc, /* other parameters */) const {/* body of the kernel */}
}
```
The qualifier `ALPAKA_FN_ACC` is equivalent to cuda `__global__` or `__device__`. The first, mandatory parameter for an alpaka kernel is the accelerator, on whose type the kernel call is internally templated. 

While kernels are usually called in the `.cc` and `.cu` files respectively to run on CPU and on GPU, alpaka kernels are called directly in the `.cc` file through the `alpaka::enqueue` function.

### Work division and loops
Similarly to cuda, where the number of blocks and the number of threads per block are given to kernel calls through the `<<<...>>>` syntax, alpaka kernels need a **valid work division**, that can be configured through `cms::alpakatools::make_workdiv<AccND>(blocksPerGrid, threadsPerBlockOrElementsPerThread)`. `N` in the parameter template indicates the  needed dimensionality for the operations that the kernel will execute.

The second parameter of the `make_workdiv` function is the block size on GPU or the elements per threads on CPU, as alpaka provides an additional abstraction level with respect to CUDA. Generally, a loop over the threads in CUDA is equivalent to two loops in alpaka, an outer loop over the threads and an internal loop over the elements of the thread. The function `for_each_element_in_block_strided` helps to write this kind of loops in an easier way:

#### CUDA
```c++
for (auto j = threadIdx.x; j < sampleVec.size(); j += blockDim.x) { sampleVec[j] = 0; }
```
#### ALPAKA
```c++
cms::alpakatools::for_each_element_in_block_strided(acc, sampleVec.size(), [&](uint32_t j) { sampleVec[j] = 0; })
```
Variants of this helper function exist, i.e. to loop with strided access.

:::info
When configuring the work division, the helper function **`cms::alpakatools::divide_up_by(dataSize, blockSize)`** can be used to calculate the number of blocks as *(dataSize + blockSize - 1)/blockSize*.
:::

### Launching the kernel
The `alpaka::enqueue` function takes two arguments:
- an alpaka `queue`
- a `taskKernel`, which is responsible for the kernel run and is created through `alpaka::createTaskKernel<AccND>(workDiv, kernelName, kernelParameters)` 

When launching the kernel, it is not necessary to pass the accelerator parameter in the `taskKernel`, as it provided automatically due to the queue.

### Usage example

``` c++
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDiv.h"

blockSize = 64;
numberOfBlocks = 8;
const workDiv1D myWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(myWorkDiv, myAlpakaKernel(), /* kernel parameters */));
```

## OTHER

| CUDA / Serial                               | ALPAKA                                                              |
|:------------------------------------------- |:------------------------------------------------------------------- |
| `cms::cuda::copyAsync(ptr, object, stream)` | `alpaka::memcpy(queue, buf_device, buf_host)`                       |
| `__forceinline__` / `inline`                | `ALPAKA_FN_INLINE`                                                  |
| `atomicFoo(args)`                           | `alpaka::atomicFoo(acc, args, alpaka::hierarchy::Grid/Blocks/Threads{})` |
| `cms::cuda::currentDevice()`                | `cms::alpakatools::getDeviceIndex(alpaka::getDev(queue))`           |

For information about cuda/hip equivalent functions in alpaka, additional coding guidelines and more, please consult [the latest documentation](https://alpaka.readthedocs.io/en/latest/index.html).