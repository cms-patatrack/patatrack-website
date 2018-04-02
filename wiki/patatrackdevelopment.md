---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack
The current Patatrack development branch is based on `CMSSW_10_1_0`, and uses the `slc7_amd64_gcc630` architecture.  
For a different branch and architatcure, adapt these instructions as needed.

### Create a local working area as usual
```bash
export SCRAM_ARCH=slc7_amd64_gcc630
cmsrel CMSSW_10_1_0
cd CMSSW_10_1_0/src/
cmsenv
git cms-init
```

### Optional: setup the NVIDIA drivers
CMSSW is set up to pick up the NVIDIA drivers and CUDA runtime from the host machine.
If the machine you are using has one or more NVIDIA GPUs with CUDA 9.1 already installed, you don't need to do anything to use them.

If the machine you are using *does not have* a GPU with the NVIDIA drivers and CUDA runtime, set them up in CMSSW:
```bash
modprobe -n -q nvidia || scram setup nvidia-drivers
```

### Build the CUDA code
The standard releases do not build the CUDA-related code (yet); check it out and build it:
```bash
git cms-addpkg HeterogeneousCore
scram b
```

### Check out the patatrack development branch
Add the patatrack repository and create a development branch based on the Patatrack one:
```bash
git cms-remote add cms-patatrack
git checkout cms-patatrack/CMSSW_10_1_X_Patatrack -b my_development
```

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

## Special instructions for working on `vinavx2`
On `vinavx2` a special CMSSW release is available, `CMSSW_10_1_0_Patatrack`, built from the `cms-patatrck/cmssw` repository including the current `CMSSW_10_1_X_Patatrack` branch on top of `CMSSW_10_1_0`.  
To create a local working area, one can do
```bash
source /data/cmssw/cmsset_default.sh
cmsrel CMSSW_10_1_0_Patatrack
cd CMSSW_10_1_0_Patatrack/src
cmsenv
cp -ar $CMSSW_RELEASE_BASE/git .git
git checkout -- .clang-tidy .gitignore
```

From here one can work in the `CMSSW_10_1_X_Patatrack`, or create a new topic branch with the usual commands:
```bash
git checkout -b my_development
```

## Create a pull request
  - open https://github.com/cms-patatrack/cmssw

  - there should be box with the branch you just created and a green button saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")

  - click on it, and create a pull request as usual:
    ![Create a pull request](screenshot2.png "Create a request")

