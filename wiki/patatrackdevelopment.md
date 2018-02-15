---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack
The current Patatrack development branch is based on `CMSSW_10_1_0_pre1`, and uses the `slc7_amd64_gcc630` architecture.  
For a different branch and architatcure, adapt these instructions as needed.

### Create a local working area as usual
```bash
export SCRAM_ARCH=slc7_amd64_gcc630
cmsrel CMSSW_10_1_0_pre1
cd CMSSW_10_1_0_pre1/src/
cmsenv
git cms-init
```

### Setup the NVIDIA drivers
CMSSW is set up to pick up the NVIDIA drivers and CUDA runtime from the host machine.
If the machine you are using has one or more NVIDIA GPUs with CUDA 9.1 already installed, you don't need to do anything to use them.

If the machine you are using *does not have* a GPU with the NVIDIA drivers and CUDA runtime, set them up in CMSSW:
```bash
scram setup nvidia-drivers
```

### Check out the patatrack and development branch
Add the patatrack repository and create a develpmt branch based on the Patatrack one:
```bash
git cms-remote add cms-patatrack
git checkout cms-patatrack/CMSSW_10_1_X_Patatrack -b my_development
```

### Optional: fix `nvcc.profile` to let `nvcc` work on the command line
```bash
cmsenv
eval $(scram tool info cuda | grep ^CUDA_BASE)
rm -f $CMSSW_BASE/external/$SCRAM_ARCH/bin/nvcc.profile
cat > $CMSSW_BASE/external/$SCRAM_ARCH/bin/nvcc.profile << @EOF

TOP              = $CUDA_BASE

NVVMIR_LIBRARY_DIR = \$(TOP)/nvvm/libdevice

LD_LIBRARY_PATH += \$(TOP)/lib:
PATH            += \$(TOP)/nvvm/bin:\$(TOP)/bin:

INCLUDES        +=  "-I\$(TOP)/include" \$(_SPACE_)

LIBRARIES        =+ \$(_SPACE_) "-L\$(TOP)/lib\$(_TARGET_SIZE_)/stubs" "-L\$(TOP)/lib\$(_TARGET_SIZE_)"

CUDAFE_FLAGS    +=
PTXAS_FLAGS     +=
@EOF
```

### No longer needed: make `cuda-api-wrappers` available
The `cuda-api-wrappers` external is already avaliable in the CMSSW 10.1.x

### Check out the modified packages and their dependencies
```bash
git cms-addpkg $(git diff $CMSSW_VERSION --name-only | cut -d/ -f-2 | sort -u)
git cms-checkdeps -a
```

### Write code, compile, debug, commit, and push to your repository
```bash
...
scram b
...
git push my-cmssw HEAD:my_development
```

### Create a pull request
  - open https://github.com/cms-patatrack/cmssw

  - there should be box with the branch you just created and a green button saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")

  - click on it, and create a pull request as usual:
    ![Create a pull request](screenshot2.png "Create a request")

