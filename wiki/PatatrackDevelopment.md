---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack
The current Patatrack development branch is based on `CMSSW_10_5_X`, and supports the architecture `slc7_amd64_gcc700`.


## Installing `CMSSW_10_5_0_pre2_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_5_0_pre2`, using this dedicated release has few advantages:
  - update CUDA to version 10.1.105;
  - include the changes from the `CMSSW_10_5_X_Patatrack` development branch.

On **vinavx2** the release is available after `source /data/cmssw/cmsset_default.sh`.
On **cmg-gpu1080** the release is available after `source /data/patatrack/cmssw/cmsset_default.sh`.

Otherwise, see [the instructions](PatatrackReleases.md) for installing `CMSSW_10_5_0_pre2_Patatrack` on your machine.


### Older CUDA versions
`CMSSW_10_5_0_pre2` has been built with CUDA 10.1.105, that requires NVIDIA drivers version 418.39 or newer.
To support older systems, two alternate builds are available:
  - `CMSSW_10_5_0_pre2_Patatrack_CUDA_10_0`, built with CUDA version 10.0.130, requiring drivers version 410.48 or newer;
  - `CMSSW_10_5_0_pre2_Patatrack_CUDA_9_2`, build with CUDA version 9.2.148 Update 1, requiring drivers version 396.37 or newer.

They can be installed (see above) and used (see below) in the usual way.


## Create a working area for `CMSSW_10_5_0_pre2_Patatrack`

### Source the local installation
Source the script `cmsset_default.sh` in the directory where you have installed the Patatrack releases, e.g.:

```bash
export SCRAM_ARCH=slc7_amd64_gcc700
source /data/cmssw/cmsset_default.sh
```


### Set up a working area
```bash
scram list CMSSW_10_5_0_pre2
cmsrel CMSSW_10_5_0_pre2_Patatrack
cd CMSSW_10_5_0_pre2_Patatrack/src
cmsenv
```


### Set up the local `git` repository
```bash
git cms-init -x cms-patatrack
git branch CMSSW_10_5_X_Patatrack --track cms-patatrack/CMSSW_10_5_X_Patatrack
```

You should be able to work in the `from-CMSSW_10_5_0_pre2_Patatrack` branch as you would in a normal CMSSW development area.


### Check out the patatrack development branch
To work on further developments, it is advised to start from the HEAD of the patatrack branch.

```bash
cmsenv
git checkout cms-patatrack/CMSSW_10_5_X_Patatrack -b my_development_branch
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
git push -u my-cmssw HEAD:my_development_branch
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

  - make sure to choose `CMSSW_10_5_X_Patatrack` as the target branch, **not** the `master` branch
