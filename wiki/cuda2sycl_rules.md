# CUDA to SYCL/OneAPI rules

> **_NOTE:_**  the examples shown in the tables imply ```using namespace sycl```, while this is not true for code snippets. At the beginning of each section, if needed, there is a box with CUDA definitions and their SYCL equivalent.

## SYCL Setup
1. Install the Intel OneAPI DPCPP SYCL backend following the official instructions from [Intel oneAPI Base Kit](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html). Tip: For Linux, package guided installation is easy. For WSL, you need to follow Linux installation.
2. For compiling and running in Linux, the toolkit local variables need to be set on each new terminal session with the following command: ```. /opt/intel/oneapi/setvars.sh```
3. Try compiling a C++ file including sycl header files (```#include <CL/sycl.hpp>```) with the command: ```dpcpp <source_file> -o <name_of_executable>```
4. Everything is ready! 
5. (optional) If you want to run code on an integrated Intel GPU while using WSL, you'll need a specific set of drivers and an additional installation package which can be found in [this repository](https://github.com/intel/compute-runtime/blob/master/WSL.md).

## Execution model

|            CUDA            |          SYCL           |
|:--------------------------:|:-----------------------:|
| Stream Multiprocessor (SM) |    Compute Unit (CU)    |
|          SM core           | Processing Element (PE) |
|           thread           |       work - item       |
|           block            |      work - group       |
|            grid            |        ND-range         |

The main difference between CUDA and SYCL is that the former has been developed specifically for GPUs, while the latter is oriented toward a broader set of architectures. The main goal of SYCL is to write the code once and being able to execute it on different backends. For this reason the processing elements (PE) in SYCL can represent other hardware on non-GPU architectures. This is implementation-specific, but allows for a conceptual equivalence between PEs and SM cores.

Diving into the execution model, the main element in a SYCL thread hierarchy is the ND-range, that is a 1, 2 or 3 dimensional grid of work groups, that are equally sized groups of the same dimensionality. These work groups are in turn divided into work items. 

- Work item : a single thread within the thread hierarchy.
- Work group : a 1, 2 or 3 dimensional set of threads (work-items) within the thread hierarchy. 
- ND-range : it's the representation of the thread hierarchy used by the SYCL runtime when executing work. It has three components: the global range, the local range and the number of work groups. The global range is the total number of work items within the thread hierarchy in each dimension; the local range is the number of work items in each work group in the thread hierarchy, and the number of work groups is the total number of work groups in the thread hierarchy, which can be derived from the global and local.

On some modern hardware platforms there is also the possibility of accessing subsets of the work-items in a
work-group to execute them with additional scheduling guarantees. This subsets are known as sub-groups. 

In the following figure there is a representation of these concepts:
![](https://codimd.web.cern.ch/uploads/upload_10a867007d9836f796d09c66fb4cddc6.png)

In SYCL, work-group size can be left empty and then implementation can set it to the optimal value according to an internal heuristic. Additionally, SYCL does provide a mechanism to get the preferred work-group size. An example on how to do it is shown below. Typically, choosing a work group size that is a multiple of the preferred one will be enough.
```cpp
auto preferredWorkGroupSizeMultiple = kernel.get_work_group_info<
sycl::info::kernel_work_group::preferred_work_group_size_multiple>();

queue.submit([&](sycl::handler &cgh){
    cgh.parallel_for(kernel, 
                     sycl::range<1>(preferredWorkGroupSizeMultiple), 
                     [=](sycl::id<1> idx){
        /* kernel code */
      });
  });
```
Finally, to see how the warp concept is mapped on work group sizes it's possible to consult the SYCL or OpenCL implementation notes for the given GPU architecture. Any assumption of warp execution is not performance portable (although developers can select different kernels depending on the underlying architecture). 

### Kernels execution

|               CUDA                |               SYCL                |
|:---------------------------------:|:---------------------------------:|
|            <<<....>>>             |          nd_range class           |
| "kernel function name"<<<...>>>() |          queue::submit()          |
|                N/A                |           nd_item class           |
|         gridDim.{x, y, z}         |  nd_item::get_group_range({0,1,2})  |
|        blockDim.{x, y, z}         | nd_item::get_local_range({0,1,2}) |
|        blockIdx.{x, y, z}         |    nd_item::get_group({0,1,2})    |
|        threadIdx.{x, y, z}        |  nd_item::get_local_id({0,1,2})   |
|                N/A                |  nd_item::get_global_id({0,1,2})  |
|                N/A                |  nd_item::get_linear_group_id()   |
|                N/A                |  nd_item::get_linear_local_id()   |
|                N/A                |  nd_item::get_linear_global_id()  |

SYCL kernel functions are called using one of the following invoke API entries (which are methods of SYCL handlers):

- ```single_task```: The kernel function is executed exactly once. For example:
```cpp
auto queue = sycl::queue(sycl::default_selector{})
queue.submit([&](sycl::handler& h)
             {
                 h.single_task([=]{a[0] = 1.0f});
             });
```
- ```parallel_for```: The kernel function is executed ND-range times passing thread identification objects as parameters. For example:
```cpp
auto queue = sycl::queue(sycl::default_selector{})
queue.submit([&](sycl::handler& h)
             {
                h.parallel_for(range, [=](id<1> i) {a[i] = b[i]}); 
             });
```

The handler defines the interface to invoke kernels by submitting commands to a queue.
A handler can only be constructed by the SYCL runtime and is passed as an argument to the command group function. The command group function is an argument to submit.

The SYCL kernel function invoke API takes a C++ callable object by value which is most often expressed as a lambda. If local memory is required or work-group size is specified manually, then the corresponding ```nd_range object``` must be used as first parameter. In turn, the nd_item associated with the nd_range can be passed inside the kernel and its method for barriers or work-group operation can be used.
In the following example we demonstrate how to call a SYCL kernel in the same way that one would do using ```<<<gridSize,blockSize>>>``` in CUDA:
```cpp
// CUDA version
const dim3 blockSize(numThreadsPerBlock, 1, 1);
const dim3 gridSize(numBlocks, 1, 1);
 
kernel<<<gridSize, blockSize>>>(d_input, d_output);

cudaDeviceSynchronize();

// SYCL version
auto queue = sycl::queue(sycl::default_selector{});
const sycl::range<3> blockSize(numThreadsPerBlock, 1, 1);
const sycl::range<3> gridSize(numBlocks, 1, 1);

queue.submit([&](sycl::handler& cgh) 
{
  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), 
                                     [=](sycl::nd_item<3> item)
    {
        kernel(d_input, d_output, item);
    });
}).wait();
```
Kernel functions qualifiers like ```__global__, __shared__``` are not needed in SYCL as they are all abstracted by the SYCL runtime classes.

Another important difference is found when passing values to a funcition inside a ```queue::submit```. As per SYCL specification, variables can be passed inside a kernel (```parallel_for```) only by value, moreover the capture of ```*this``` is not allowed neither implicitly nor explicitly (in general, ```this``` would point to host memory which is not accessible on the device). To resolve this isssue you can simply create local copies of all the variables needed inside the kernel before the ```parallel_for``` gets called, as demonstrated in the following example:
```cpp
sycl::queue q = sycl::queue(sycl::default_selector{});
q.submit([&](sycl::handler& h)
{
    auto var_kernel = var;
    h.parallel_for(...,[=](...)
    {
        /* use var_kernel here */               
    });
});
```

### Synchronization
In CUDA, there are two synchronization levels:

- ```cudaDeviceSynchronize()```: blocks the threads call until all the work on the device is completed
- ```__syncthreads()```: wait for all the threads in a thread block to reach the same synchronization point

In SYCL, the only synchronization that is possible is across all work items within a work group using barriers which can be called within a kernel function:
- ```mem_fence```: inserts a memory fence on global memory access or local memory access across all work items within a work group. It's the analog of CUDA ```__threadfence_block()```.
- ```barrier```: like the previous one, but it also blocks the execution of each work item within a work group at that point until all of them have reached that point. 

We demonstrate the equivalence of CUDA and SYCL synchronization methods in the following snippet, taken from a parallel reduction kernel:

```cpp
// CUDA version
__shared__ int sdata[numThreadsPerBlock];

unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

sdata[tid] = input[i];

__syncthreads();

// SYCL version
unsigned int tid = item.get_local_id(0);
unsigned int i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

sdata[tid] = input[i];

item.barrier();
```

SYCL does not provide any memory fences or barriers across the entire kernel, only across the work items within a work group.

> **_NOTE:_**  In the example we show how to get the corresponding values of the indices ```tid``` and ```i```. While in CUDA the kernel has direct access to information about the number of threads and dimension of blocks, in SYCL you have to explicitly pass a ```nd_item``` corresponding to the ```nd_range``` which is the argument of the ```parallel_for```.

SYCL queues are similar, in principle, to CUDA streams. One key difference is found in the default implementation. In particular, SYCL queues are by default not in order, meaning that different kernels (i.e. ```queue::submit()```) can and will be executed at the same time whenever possible to improve performances.
If the data used by different kernels is not independent, you should initialize a queue as in order passing an extra argument in the constructor as shown below:

```cpp
// Initialize an out-of-order queue
auto not_in_order_q = sycl::queue(sycl::default_selector{});

// Initialize an in-order queue
auto in_order_q = sycl::queue(sycl::default_selector{}, sycl::property::queue::in_order());
```
Using an in-order queue corresponds exactly to adding a ```queue::wait()``` after each and every submit. In this way the queue behaves as the Stream '0' in CUDA: operations are exectuded in issue-order on the selected device.
In case the code allows for some kernels to be executed in parallel while others have explicit dependencies, it is possible to use a default queue and specify the desired data dependencies using the member function ```handler::depends_on()``` which can take an event or an array of events as parameters. This is demonstrated in the example below:
```cpp
// Initialize an out-of-order queue
auto q = sycl::queue(sycl::default_selector{});
// Define an event
auto e = q.submit([&](sycl::handler& h)
                  {
                      /* kernel */
                  }
q.submit([&](sycl::handler& h)
         {
             h.depends_on(e);
             /* kernel */
         });
q.submit([&](sycl::handler& h)
         {
             /* kernel */
         })
```
In this case, the second submit has an explicit dependency on the first, so the queue will wait and syncronize before executing it. However, the last submit can be executed as the same time as the first one since no data dependency is specified.

## Memory management

|                  CUDA                   |         SYCL          |
|:---------------------------------------:|:---------------------:|
|   Per-thread local memory / register    |    private memory     |
| Per-block shared memory / shared memory |     local memory      |
|             constant memory             |    constant memory    |
|              global memory              |     global memory     |
|             Texture memory              |     Image memory      |
|              local memory               | N/A (device specific) |

SYCL is based on the OpenCL memory model but operates at a higher level of abstraction, which means that storage and access memory are separated and treated with different objects: buffers and accessors, respectively. SYCL buffers are, at their core, std::unique_ptr wrapped in such a way to make them live only inside the scope they are defined in. The buffer manages memory allocation/copy, while accessors create requirements on the buffers. These requirements can be allocating memory, synchronization between different accessors or data transfer between host and device. Depending on the accessor, data is automatically allocated on the host or on the device.
However, it is possible to explicitly allocate memory either on the host, the device or shared between the two using C-like pointers and provided that the device supports Universal Shared Memory (USM).
It is important to distinguish the different types of memory in SYCL which differ somewhat from the ones used in CUDA:
- Private memory: region of memory allocated per work item and only visible to that work item. Cannot be accessed from the host.
- Local memory: contiguous region of memory allocated per work group and visible to all of the work items in that work group. This memory is allocated and accessed using an accessor and cannot be accessed from the host.
- Global memory: pool of memory visible by all of the work groups of the device.

We note that the explicit allocation of memory through pointers doesn’t allow specifying whether it should be local or private and defaults to private. At this time, the only way to allocate local memory is through the use of an accessor.

> **_NOTE:_** Although the methods are really similar, shared memory has different meanings in CUDA and SYCL. SYCL shared memory creates a single pointer which can be accessed both by the host and the device and allows to avoid the explicit copying of data between the two on supported devices. Shared memory in CUDA instead, is "shared" between the threads of a block.

## Memory allocation and transfer
|                 CUDA                 |            SYCL            |
|:------------------------------------:|:--------------------------:| 
|             cudaMalloc()             |      malloc_device()       |
|             cudaMemset()             |          memset()          |
|             cudaMemcpy()             |          memcpy()          |
|              cudaFree()              |           free()           |

Regarding the allocation of memory on the device, the migration from CUDA to SYCL is pretty simple. 
In CUDA there are:

- ```cudaMalloc()```: this function allocates a device pointer on the device and returns the address of the allocated memory. Then the memory can be initialized with ```cudaMemset()```, copied with ```cudaMemcpy()``` and freed with ```cudaFree()```.
- ```cudaMemcpyAsync```: used for the asynchronous version of data transfer. It is non-blocking with respect to the host, so calling the function may return before completing the copy.

On the other hand, in SYCL we have:

- ```malloc_device()```: device allocations that can be read from or written to by kernels running on a device, but they cannot be directly accessed from code executing on the host. Data must be copied between host and device using the explicit USM memcpy mechanisms.
- ```free()```: free memory allocated with *Malloc* functions
- ```memcpy()```: copy memory between host and device
- ```memeset()```: set memory allocated with *Malloc* functions 
- ```copy()```: copy memory from src to dest
- ```fill()```: fill the destination with values passed

The last four methods are available also as members of the class ```sycl::handler```, so they can be used in the scope of the queue.
An example is shown below:
```cpp
Q.submit([&](handler &h) {
    // copy hostArray to deviceArray
    h.memcpy(device_array, &host_array[0], N * sizeof(int));
    });
Q.wait();
```

In the following example we show the explicit allocation, setting, copy and deallocation of memory first in CUDA and then in its equivalend SYCL form:
```cpp
// CUDA version
std::vector<float> h_a(N);
float* d_a;
float* d_b;

cudaMalloc(&d_a, N * sizeof(float));
cudaMalloc(&d_b, N * sizeof(float));

cudaMemset(d_a, 0x00, N * sizeof(float));
cudaMemset(d_b, 0x00, N * sizeof(float));

cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, d_a, N * sizeof(float), cudaMemcpyDeviceToDevice);
cudaMemcpy(h_a.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(d_a);
cudaFree(d_b);

// SYCL version
std::vector<float> h_a(N);
auto queue = sycl::queue{sycl::default_selector{}};
auto d_a = sycl::malloc_device<float>(N, queue);
auto d_b = sycl::malloc_device<float>(N, queue);

queue.memset(d_a, 0x00, N * sizeof(float));
queue.memset(d_b, 0x00, N * sizeof(float));

queue.memcpy(d_a, h_a.data(), N * sizeof(float));
queue.memcpy(d_b, d_a, N * sizeof(float));
queue.memcpy(h_a.data(), d_b, N * sizeof(float)).wait();

sycl::free(d_a, queue);
sycl::free(d_b, queue);
```
We note that in order to allocate in SYCL it is necessary to have declared a queue first, as to choose the device on which the memory should be allocated and data copied. Moreover, since copying is asynchronous by default, it is a good practice to always use ```sycl::queue::wait()``` after any group of copying actions to prevent segmentation faults or unexpected behaviours. Finally, in SYCL it is not necessary to specify the direction of copying (host-device, device-device, device-host) as it is deduced at runtime.

The allocation of memory on the host can be done with C++ methods in both cases, but it will be virtual memory. CUDA allows us to allocate pinned host memory with specific methods, while SYCL runtime manages it on its own aiming at allocating memory in the most optimal way, so there is no explicit method to do it. However, users can manually allocate pinned memory on the host and hand it over to the SYCL implementation. This will often involve allocating host memory with a suitable alignment and multiple, and sometimes can be managed manually using OS specific operations such as mmap and munmap.
Finally, in SYCL there is no explicit mechanism to request zero-copy. If memory is allocated in pinned memory as described above, then the SYCL runtime will attempt to initialize with zero-copy if possible.

The CUDA methods are:

- ```cudaMallocHost()```: pinned host memory. A dev_ptr is passed to it and it is the host pointer accessible from the device. This memory is directly accessible from the device.
- ```cudaHostAlloc()```: pinned host memory mapped into the device address space (zero-copy). It will avoid the explicit data movement between host and device. Although zero-copy improves the PCIe transfer rates, it is required to be synchronized whenever data is shared between host and device. 
- ```cudaHostAllocPortable``` that allows to allocate memory that will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
-```cudaHostGetDevicePointer```: this function provides the device pointer for mapped, pinned memory.

In SYCL there are:

- ```malloc_host()```: allocate memory on the host that is accessible on both the host and the device. These allocations cannot migrate to the device’s attached memory so kernels that read from or write to this memory do it remotely, often over a slower bus such as PCI-Express.
- ```malloc_shared()```: like host allocations, shared allocations are accessible on both the host and device, but they are free to migrate between host memory and device attached memory automatically.

The following is an example for host allocation, taking into account that the two methods used **don't** do exactly the same thing.

```cpp
// CUDA version
float* hostPtr;
cudaMallocHost(&hostPtr, N * sizeof(float));
for (int i = 0; i < N; i++) {
// Initialize hostArray on host
    hostPtr[i] = i;
}

// SYCL version
sycl::queue Q;
int *host_array = sycl::malloc_host<int>(N, Q);
for (int i = 0; i < N; i++) {
// Initialize hostArray on host
    host_array[i] = i;
}
```

Summary of SYCL malloc methods:
| FUNCTION CALL | DESCRIPTION                                            | ACCESSIBLE ON HOST | ACCESSIBLE ON DEVICE |
| :-------------: | :------------------------------------------------------: | :------------------: | :--------------------: |
| malloc_device | Allocation on device, explicit data movement           | NO                 | YES                  |
| malloc_host   | Allocation on host, implicit data movement             | YES                | YES                  |
| malloc_shared | Shared allocation, can migrate between host and device, implicit data movement | YES        | YES  |

## References
[codeplay CUDA to SYCL](https://developer.codeplay.com/products/computecpp/ce/guides/sycl-for-cuda-developers)

CUDA

  - [Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming) 

SYCL

  - [Documentation](https://sycl.readthedocs.io/en/latest/index.html)
  - [Khronos doc](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html)
  - [Data parallel C++](https://link.springer.com/content/pdf/10.1007%2F978-1-4842-5574-2.pdf)

OneAPI

  - [Developer guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top.html)
  - [GPU optimization guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top.html)