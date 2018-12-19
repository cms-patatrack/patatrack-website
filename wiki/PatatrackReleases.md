---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing `CMSSW_10_4_0_pre4_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_4_0_pre4`, using this dedicated release has few advantages:
  - update Eigen and improve compatibility with CUDA:
    - extend support for self-adjoint matrices in CUDA code;
  - enable `cub` error reporting for CUDA library calls;
  - include the changes from the CMSSW_10_4_X_Patatrack development branch.

If you are working on **vinavx2** you can skip these steps, as the release is already available.

### Bootstrap a local installation of CMSSW
Choose for `VO_CMS_SW_DIR` a directory to which you have write permissions, and has at least 20 GB of spare disk space.  
**Do not** use a directory on EOS.

```bash
cd <...>
export VO_CMS_SW_DIR=$PWD
export SCRAM_ARCH=slc7_amd64_gcc700
export LANG=C
wget http://cmsrep.cern.ch/cmssw/bootstrap.sh -O $VO_CMS_SW_DIR/bootstrap.sh
chmod a+x $VO_CMS_SW_DIR/bootstrap.sh
$VO_CMS_SW_DIR/bootstrap.sh -a slc7_amd64_gcc700 -r cms -path $VO_CMS_SW_DIR setup
```

### Install `CMSSW_10_4_0_pre4` and its dependencies
Most of the externals for `CMSSW_10_4_0_pre4_Patatrack` need to be installed from the official repository; the easiest approach is to install them automatically together with `CMSSW_10_4_0_pre4`:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 upgrade -y
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 install -y cms+cmssw+CMSSW_10_4_0_pre4
```

### Install `CMSSW_10_4_0_pre4_Patatrack`
Patatrack releases can now be installed by `cmspkg`, using the dedicated repository:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -r cms.patatrack install -y cms+cmssw+CMSSW_10_4_0_pre4_Patatrack
```
