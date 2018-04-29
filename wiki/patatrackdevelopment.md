---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack
The current Patatrack development branch is based on `CMSSW_10_2_X`, and uses the `slc7_amd64_gcc630` architecture.  
For a different branch and architatcure, adapt these instructions as needed.  

Starting with `CMSSW_10_2_X_2018-04-29-0000`, the architecture `slc7_amd64_gcc700` is also tentaviely supported.  
`CMSSW_10_2_0_pre2` can be used with GCC 7.x following the instructions below.

### Create a local working area as usual
```bash
export SCRAM_ARCH=slc7_amd64_gcc630
cmsrel CMSSW_10_2_0_pre2
cd CMSSW_10_2_0_pre2/src/
cmsenv
git cms-init
```

### Update SCRAM and CUDA to support GCC 7.x
Starting with `CMSSW_10_2_X_2018-04-29-0000`, the architecture `slc7_amd64_gcc700` is also tentaviely supported.  
If using `CMSSW_10_2_0_pre2`, update it to be compatible with GCC 7.x:
```bash
curl -s http://fwyzard.web.cern.ch/fwyzard/patatrack/CMSSW_10_2_0_pre2/config.tgz | tar xz -C $CMSSW_BASE
cmsenv
scram b -j0
```

### Check out the patatrack development branch
Add the patatrack repository and create a development branch based on the Patatrack one:
```bash
git cms-remote add cms-patatrack
git checkout cms-patatrack/CMSSW_10_2_X_Patatrack -b my_development
```

### Check out the modified packages and their dependencies
```bash
git cms-addpkg $(git diff $CMSSW_VERSION --name-only | cut -d/ -f-2 | sort -u)
git cms-checkdeps -a
```

### Write code, compile, debug, commit, and push to your repository
```bash
...
scram b -j`nproc`
...
git push my-cmssw HEAD:my_development
```

## Special instructions for working on `vinavx2` with `CMSSW_10_1_0_Patatrack`
<details><summary>Out of date - click to show
</summary>

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
</details>

## Create a pull request
  - open https://github.com/cms-patatrack/cmssw

  - there should be box with the branch you just created and a green button saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")

  - click on it, and create a pull request as usual:
    ![Create a pull request](screenshot2.png "Create a request")

