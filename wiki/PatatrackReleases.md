---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing `CMSSW_10_5_0_pre1_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_5_0_pre1`, using this dedicated release has few advantages:
  - extend Eigen's support for matrix decomposition (QR, LLT, LDLT) with CUDA
  - include the changes from the CMSSW_10_5_X_Patatrack development branch.

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

### Install `CMSSW_10_5_0_pre1` and its dependencies
Most of the externals for `CMSSW_10_5_0_pre1_Patatrack` need to be installed from the official repository; the easiest approach is to install them automatically together with `CMSSW_10_5_0_pre1`:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 upgrade -y
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 install -y cms+cmssw+CMSSW_10_5_0_pre1
```

### Install `CMSSW_10_5_0_pre1_Patatrack`
Patatrack releases can now be installed by `cmspkg`, using the dedicated repository:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -r cms.patatrack install -y cms+cmssw+CMSSW_10_5_0_pre1_Patatrack
```
