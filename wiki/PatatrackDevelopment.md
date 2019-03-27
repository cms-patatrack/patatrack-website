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


## Installing `CMSSW_10_5_0_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_5_0`, using this dedicated release has few advantages:
  - include the changes from the `CMSSW_10_5_X_Patatrack` development branch.

On **vinavx2** the release is available after `source /data/cmssw/cmsset_default.sh`.
On **cmg-gpu1080** the release is available after `source /data/patatrack/cmssw/cmsset_default.sh`.

Otherwise, see [the instructions](PatatrackReleases.md) for installing `CMSSW_10_5_0_Patatrack` on your machine.


## Create a working area for `CMSSW_10_5_0_Patatrack`

### Source the local installation
Source the script `cmsset_default.sh` in the directory where you have installed the Patatrack releases, e.g.:

```bash
export SCRAM_ARCH=slc7_amd64_gcc700
source /data/cmssw/cmsset_default.sh
```


### Set up a working area
```bash
scram list CMSSW_10_5_0
cmsrel CMSSW_10_5_0_Patatrack
cd CMSSW_10_5_0_Patatrack/src
cmsenv
```


### Set up the local `git` repository
```bash
git cms-init -x cms-patatrack
git branch CMSSW_10_5_X_Patatrack --track cms-patatrack/CMSSW_10_5_X_Patatrack
```

You should be able to work in the `from-CMSSW_10_5_0_Patatrack` branch as you would in a normal CMSSW development area.


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
