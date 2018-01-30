---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack
These instruction assume that the Patatrack development branch is based on `CMSSW_10_0_0`, and use the `slc7_amd64_gcc630` architecture.  
For a different branch and architatcure, adapt them as needed.

### Create a local working area as usual
```
export SCRAM_ARCH=slc7_amd64_gcc630
cmsrel CMSSW_10_0_0
cd CMSSW_10_0_0/src/
cmsenv
git cms-init
```

### Check out the patatrack and development branch
Add the patatrack repository and create a develpmt branch based on the Patatrack one:
```
git cms-remote add cms-patatrack
git checkout cms-patatrack/CMSSW_10_0_0_Patatrack -b my_development
```

### Check out the modified packages and their dependencies
```
git cms-addpkg $(git diff $CMSSW_VERSION --name-only | cut -d/ -f-2 | sort -u)
git cms-checkdeps -a
```

### Optional: make `cuda-api-wrappers` available
The `cuda-api-wrappers` external is avaliable in the CMSSW 10.1.x IBs, but not in 10.0.0.
To make it available in your local area, run
```
scram setup /cvmfs/cms-ib.cern.ch/week1/slc7_amd64_gcc630/cms/cmssw-tool-conf/41.0/tools/selected/cuda-api-wrappers.xml
```

### Write code, compile, debug, commit, and push to your repository
```
...
scram b
...
git push my-cmssw HEAD:my_development
```

### Create a pull request
  - open https://github.com/cms-patatrack/cmssw

  - there should be box with the branch you just created and a green button saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")

  - click on it, and create a pull request as usual
    ![Create a pull request](screenshot2.png "Create a request")

