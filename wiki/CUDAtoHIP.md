# Table Comparing Syntax for Different Compute APIs

|Term|CUDA|HIP|
|---|---|---|
|Device|`int deviceId`|`int deviceId`|
|Queue|`cudaStream_t`|`hipStream_t`|
|Event|`cudaEvent_t`|`hipEvent_t`|
|Memory|`void *`|`void *`|
||||
| |grid|grid|
| |block|block|
| |thread|thread|
| |warp|warp|
||||
|Thread-<br>index | threadIdx.x | hipThreadIdx_x |
|Block-<br>index  | blockIdx.x  | hipBlockIdx_x  |
|Block-<br>dim    | blockDim.x  | hipBlockDim_x  |
|Grid-dim     | gridDim.x   | hipGridDim_x   |
||||
|Device Kernel|`__global__`|`__global__`|
|Device Function|`__device__`|`__device__`|
|Host Function|`__host_` (default)|`__host_` (default)|
|Host + Device Function|`__host__` `__device__`|`__host__` `__device__`|
|Kernel Launch|`<<< >>>`|`hipLaunchKernel`|
||||
|Global Memory|`__global__`|`__global__`|
|Group Memory|`__shared__`|`__shared__`|
|Constant|`__constant__`|`__constant__`|
||||
||`__syncthreads`|`__syncthreads`|
|Atomic Builtins|`atomicAdd`|`atomicAdd`|
|Precise Math|`cos(f)`|`cos(f)`|
|Fast Math|`__cos(f)`|`__cos(f)`|
|Vector|`float4`|`float4`|
