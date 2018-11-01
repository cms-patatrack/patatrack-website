---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack
The current Patatrack development branch is based on `CMSSW_10_2_X`, and supports the architecture `slc7_amd64_gcc700`.

## Installing `CMSSW_10_2_6_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_2_6`, using this dedicated release has few advantages:
  - include support for Volta-class GPUs (sm_70);
  - update Eigen and improve compatibility with CUDA:
    - update to the master branch as of Tue Sep 25 20:26:16 2018 +0200,
    - patch Tensorflow accordingly,
    - extend support for matrix inversion and diagonal matrices in CUDA code,
    - fix deprecation warnings for CUDA 10.0;
  - update LLVM/clang and improve compatibility with CUDA:
    - update to version 7.0.0, and update (or disable) clang-based externals;
    - add preliminary support for compiling with CUDA 10.0;
  - include the changes from the CMSSW_10_2_X_Patatrack branch.

If you are working on **vinavx2**, the release is already available.

Otherwise, see [the instructions](PatatrackReleases.md) for installing `CMSSW_10_2_6_Patatrack` on your machine.

## Create a working area for `CMSSW_10_2_6_Patatrack`

### Source the local installation
Source the script `cmsset_default.sh` in the directory where you have installed the Patatrack releases.  
On **vinavx2** this is `/data/cmssw`:

```bash
export SCRAM_ARCH=slc7_amd64_gcc700
source /data/cmssw/cmsset_default.sh
```

### Set up a working area
```bash
scram list CMSSW_10_2_6
cmsrel CMSSW_10_2_6_Patatrack
cd CMSSW_10_2_6_Patatrack/src
cmsenv
```

### Set up the local `git` repository
```bash
git cms-init --upstream-only || true
# you will see the error
#     fatal: 'CMSSW_10_2_6_Patatrack' is not a commit and a branch 'from-CMSSW_10_2_6_Patatrack' cannot be created from it
# it is expected, just follow the rest of the instructions

# add the Patatrack remote and branches
git cms-remote add cms-patatrack
git checkout CMSSW_10_2_6_Patatrack -b CMSSW_10_2_X_Patatrack
git branch -u cms-patatrack/CMSSW_10_2_X_Patatrack
git checkout CMSSW_10_2_6_Patatrack -b from-CMSSW_10_2_6_Patatrack

# enable the developer's repository
git cms-init
```

Now you should be able to work in the `from-CMSSW_10_2_6_Patatrack` branch as you would in a normal CMSSW development area.

### Setting up CUDA 10.0 (optional)
The Patatrack release should be compatible with CUDA 10.0.  
If it was installed, it can be set up with:
```bash
ls $VO_CMS_SW_DIR/$SCRAM_ARCH/external/cuda-toolfile/2.1-patatrack3/etc/scram.d/*.xml | xargs -n1 scram setup
```

On **vinavx2**:
```bash
ls /data/cmssw/slc7_amd64_gcc700/external/cuda-toolfile/2.1-patatrack3/etc/scram.d/*.xml | xargs -n1 scram setup
```

### Check out the patatrack development branch
To work on further developments, it is advised to start from the HEAD of the patatrack branch.

```bash
cmsenv
git checkout cms-patatrack/CMSSW_10_2_X_Patatrack -b my_development
# check out the modified packages and their dependencies
git cms-addpkg $(git diff $CMSSW_VERSION --name-only | cut -d/ -f-2 | sort -u)
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
git push -u my-cmssw HEAD:my_development
```

## Create a pull request
  - before your PR can be submitted, you should run a couple of checks to make the integration process much smoother:
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

  - there should be box with the branch you just created and a green button saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")

  - click on it, and create a pull request as usual:
    ![Create a pull request](screenshot2.png "Create a request")

  - make sure to choose `CMSSW_10_2_X_Patatrack` as the target branch
  
