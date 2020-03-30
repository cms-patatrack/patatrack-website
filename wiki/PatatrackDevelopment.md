---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## "Patatrack" CMSSW releases

### `CMSSW_11_0_X_Patatrack` stable releases

The stable Patatrack release branch is based on `CMSSW_11_0_X`, and
supports both CUDA 10.1 and CUDA 10.2, with GCC 8.3.x.

`CMSSW_11_0_0_Patatrack` is available for the architecture(s)

  - `slc7_amd64_gcc820` and CUDA 10.1 Update 1.

`CMSSW_11_0_0_Patatrack_CUDA_10_2` is available for the architecture(s)

  - `slc7_amd64_gcc820` and CUDA 10.2.

### `CMSSW_11_1_X_Patatrack` development releases

The current Patatrack development branch is based on `CMSSW_11_1_X`, and 
supports CUDA 10.2 with GCC 8.3.x.

`CMSSW_11_1_0_pre5_Patatrack` is available for the architecture(s)

  - `slc7_amd64_gcc820`;
  - `slc7_aarch64_gcc820`.

### Installation area

The `CMSSW_11_1_X_Patatrack` releases are available on CVMFS.

The `CMSSW_11_0_X_Patatrack` releases are installed locally:

  - on most Patatrack machines they are installed under `/data/cmssw/`;
  - on **cmg-gpu1080** they are installed under `/data/patatrack/cmssw/`.


## Create a working area for a Patatrack 11.0.x release

The following instructions assume the `slc7_amd64_gcc820` architecture; to use a
different one simply replace the desired architecture.

### Source the local installation
Source the script `cmsset_default.sh` in the directory where you have installed
the Patatrack releases, e.g.:

```bash
export VO_CMS_SW_DIR=/data/cmssw
export SCRAM_ARCH=slc7_amd64_gcc820
source $VO_CMS_SW_DIR/cmsset_default.sh
```


### Set up a working area for `CMSSW_11_0_0_Patatrack`
```bash
# create a working area
scram list CMSSW_11_0_0
cmsrel CMSSW_11_0_0_Patatrack
cd CMSSW_11_0_0_Patatrack/src

# load the environment
cmsenv

# set up a local git repository
git cms-init -x cms-patatrack
git branch CMSSW_11_0_X_Patatrack --track cms-patatrack/CMSSW_11_0_X_Patatrack
```

You should be able to work in the `from-CMSSW_11_0_0_Patatrack` branch as you
would in a normal CMSSW development area.


### Working with CUDA 10.2
The CMSSW 11.0.0 Patatrack release is available for two different CUDA versions:

  - `CMSSW_11_0_0_Patatrack` for CUDA 10.1 Update 1;
  - `CMSSW_11_0_0_Patatrack_CUDA_10_2` for CUDA 10.2.

They should be available and usable in the same way.


## Create a working area for a Patatrack 11.1.x release

The `CMSSW_11_1_X_Patatrack` releases are available directly on CVMFS, along
with the standard CMSSW releases.


### Set up a working area for `CMSSW_11_1_0_pre5_Patatrack`
```bash
# create a working area
scram list CMSSW_11_1_0_pre5
cmsrel CMSSW_11_1_0_pre5_Patatrack
cd CMSSW_11_1_0_pre5_Patatrack/src

# load the environment
cmsenv

# set up a local git repository
git cms-init -x cms-patatrack
git branch CMSSW_11_1_X_Patatrack --track cms-patatrack/CMSSW_11_1_X_Patatrack
```

You should be able to work in the `from-CMSSW_11_1_0_pre5_Patatrack` branch as you
would in a normal CMSSW development area.


## Working with older GPUs
CUDA is configured in CMSSW to support GPUs with Kepler (e.g. Tesla K40), Pascal
(e.g. GeForce GTX 1080, Tesla P100, ...), Volta (e.g. Titan V, Tesla V100), and
Turing (e.g. RTX 2080, Tesla T4, ...) architectures.
To work with GPUs based on different architectures, one needs to reconfigure
CUDA and rebuild all CUDA code in CMSSW with:
```bash
cmsenv
cmsCudaSetup.sh
cmsCudarebuild.sh
```


## Developing with the Patatrack branch
To work on further developments, it is advised to start from the HEAD of the
`CMSSW_11_1_X_Patatrack` branch.

### Checkout the HEAD of the development branch

```bash
cmsenv
git checkout cms-patatrack/CMSSW_11_1_X_Patatrack -b my_development_branch

# check out the modified packages and their dependencies
git diff $CMSSW_VERSION --name-only --no-renames | cut -d/ -f-2 | sort -u | xargs -r git cms-addpkg
git cms-checkdeps -a
scram b -j
```


### Write code, compile, debug, commit, and push to your repository
```bash
...
scram b -j
...
git add ...
git commit
git push -u my-cmssw HEAD:my_development_branch
```


## Create a pull request

  - before your PR can be submitted, you should run a couple of checks to make
    the integration process much smoother:
    ```bash
    # recompile with debug information (for host code) and line-number information (for device code)
    scram b clean
    USER_CXXFLAGS="-g -rdynamic" USER_CUDA_FLAGS="-g -lineinfo" scram b -j
    
    # run your code under cuda-memcheck
    cuda-memcheck --tool initcheck --print-limit 1 cmsRun step3.py
    cuda-memcheck --tool memcheck  --print-limit 1 cmsRun step3.py
    cuda-memcheck --tool synccheck --print-limit 1 cmsRun step3.py
    ```
 
  - it is also possible to run more thorough, semi-automatic checks: see [Running the validation](PatatrackValidation.md)
 
  - open https://github.com/cms-patatrack/cmssw/
 
  - there should be box with the branch you just created and a green button
    saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")
 
  - click on it, and create a pull request as usual:
    ![Create a pull request](screenshot2.png "Create a request")
 
  - make sure to choose `CMSSW_11_1_X_Patatrack` as the target branch, **not**
    the `master` branch
