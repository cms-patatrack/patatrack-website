# Setting up MPS server:

In a system with GPU resource(s), create a script for setting up the MPS server:
```bash
cat > start-mps-daemon.sh << @EOF
#! /bin/bash

# set the device(s) to exclusive mode
if [ "\$CUDA_VISIBLE_DEVICES" ]; then
  sudo nvidia-smi -i \$CUDA_VISIBLE_DEVICES -c EXCLUSIVE_PROCESS
else
  sudo nvidia-smi -c EXCLUSIVE_PROCESS
fi

# start the MPS daemon
sudo nvidia-cuda-mps-control -d
@EOF
```
Exclusive Process is recommended to be used by NVidia in order to assure that the only service using the GPU is MPS, so that it is the single point of arbitration between all CUDA processes for that GPU.
Explained in [CUDA Multi-Process Service Overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf), page 6.

Also create a script to shut the MPS server down:
```bash
cat > stop-mps-daemon.sh << @EOF
#! /bin/bash

# terminate the MPS daemon
echo quit | sudo nvidia-cuda-mps-control

# reset the device(s) to default mode
if [ "\$CUDA_VISIBLE_DEVICES" ]; then
  sudo nvidia-smi -i \$CUDA_VISIBLE_DEVICES -c DEFAULT
else
  sudo nvidia-smi -c DEFAULT
fi
@EOF
```


## Profilling the CUDA sample code

For profiling and to check the impact of running a GPU workload with MPS, one can use `nvprof` from the command line, or the Visual Profiler `nvvp &`.

Create a script similar to the following, to spawn two or more child processes to prifle:
```
$ cat mps_test_run.sh 
#!/bin/bash
/usr/local/cuda/samples/0_Simple/matrixMul/matrixMul &
/usr/local/cuda/samples/0_Simple/matrixMul/matrixMul &
wait
```

Launch the profiler, instructing it to profile all child processes, and to create one `.nvvp` file for each of them:
```
nvprof --profile-child-processes -o test_mps_%p.nvvp ./mps_test_run.sh 
```

To check whether the kernels from the different processes are running in parallel or not, import all the `.nvvp` files into a single session in the Visual Profiler:
```
nvvp test_mps_*.nvvp
```

## Profilling with the Visual profiler

For being able to check it more visually, one can do it interactively from the Visual Profiler:

Open it using `nvvp &`, open a new session "File >> New Session".
Set there "Profile child processes" option in order to be able to select the script you want and in the field "File" select the script which launches more than one binary, so you can see the timeline of every child process launched by it.

## Results of the CUDA sample code:

The results of the example, which launches two simple CUDA sample code (0_Simple/matrixmul) achieved the following results (after checking in the visual profiler that indeed the kernels were being parallelized in the profiler's timeline):

The tests were done in a NVIDIA GTX1080Ti GPU (driver version: 396.54), CUDA v9.0. 

WITH MPS server ON

```
time bash mps_test_run.sh

Performance= 894.66 GFlop/s, Time= 0.147 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Performance= 930.67 GFlop/s, Time= 0.141 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS

real	0m33.848s
user	0m13.004s
sys	0m15.954s


WITH MPS server OFF

time bash mps_test_run.sh

Performance= 724.77 GFlop/s, Time= 0.181 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Performance= 1539.39 GFlop/s, Time= 0.085 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS

real	0m39.366s
user	0m19.939s
sys	0m17.048s
```
 That makes the MPS version 15% faster than the one not using MPS. Trying to add more child processes (with 8 of them) increases the gap between the two to 25%.

## Constraints

There is a constraint in pre-Volta based architecures, the number of child processes allowed to run is restricted to 16 (and also in Volta architecure, but as this has been improved in the new architecture to 48 concurrent child processes). Above that number one starts getting cudaMalloc errors by the number of additional child processes the user has run (meaning that, if you try to run 32 child processes in a Pascal based GPU, you will get 16 errors and the rest of child processes, also 16 in this case, will go on running).

The new features introduced by Volta architecure can be read in the following paper:
http://composter.com.ua/documents/Volta-Architecture-Whitepaper.pdf,
where you can search for MPS, in order to checkout improvements done in MPS in the new architecure.

For Pascal architecture based GPUs the number of child proceses allowed is 16, but since MPS was introduced in the Kepler architecure, that one and Maxwell number of child processes has not been checked, which could be even lower.
