In a system with GPU resource(s), as root to for setting up the MPS server:
```
# cat start_mps_daemon.sh 
#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
```
Also create a script to shut the MPS server down:
```
# cat stop_mps_daemon.sh 
#!/bin/bash
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 2 -c DEFAULT
```
Then for profiling and check to see the difference between running GPU workload with and without having MPS running:

One can profile using nvprof from the CLI, or use the Visual profiler `nvvp &`.
with nvprof you can profile a script similar to the following:

$ cat mps_test_run.sh 
#!/bin/bash
/usr/local/cuda/samples/0_Simple/matrixMul/matrixMul &
/usr/local/cuda/samples/0_Simple/matrixMul/matrixMul

In order to create two different child processes, which should be checked if their kernels are being parallelized or not. For that launch the profiler, and it will generate one file per child process.

nvprof --profile-child-processes -o test_mps_%p.nvvp ./mps_test_run.sh 

This will generate different files, so it`s not really convenient to check if the kernels are actually being overlapped or not.

For being able to check it more visually, one can do it interactively from the Visual Profiler:

Open it using `nvvp &`, open a new session "File >> New Session".
Set there "Profile child processes" option in order to be able to select the script you want 

