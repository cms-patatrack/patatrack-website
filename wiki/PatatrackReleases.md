---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing `CMSSW_10_5_0_pre2_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_5_0_pre2`, using this dedicated release has few advantages:
  - update CUDA to version 10.1.105;
  - include the changes from the `CMSSW_10_5_X_Patatrack` development branch.

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

### Install `CMSSW_10_5_0_pre2` and its dependencies
Most of the externals for `CMSSW_10_5_0_pre2_Patatrack` need to be installed from the official repository; the easiest approach is to install them automatically together with `CMSSW_10_5_0_pre2`:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 upgrade -y
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 install -y cms+cmssw+CMSSW_10_5_0_pre2
```

### Install `CMSSW_10_5_0_pre2_Patatrack`
Patatrack releases can now be installed by `cmspkg`, using the dedicated repository:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -r cms.patatrack install -y cms+cmssw+CMSSW_10_5_0_pre2_Patatrack
```

### Older CUDA versions
CUDA 10.1 requires recent NVIDIA drivers, version 418.39 or newer.
To support older systems, two alternate builds are available:
  - `CMSSW_10_5_0_pre2_Patatrack_CUDA_10_0`, built with CUDA version 10.0.130, requiring drivers version 410.48 or newer;
  - `CMSSW_10_5_0_pre2_Patatrack_CUDA_9_2`, build with CUDA version 9.2.148 Update 1, requiring drivers version 396.37 or newer.

They can be installed and used in the usual way:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -r cms.patatrack install -y cms+cmssw+CMSSW_10_5_0_pre2_Patatrack_CUDA_10_0
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -r cms.patatrack install -y cms+cmssw+CMSSW_10_5_0_pre2_Patatrack_CUDA_9_2
```
