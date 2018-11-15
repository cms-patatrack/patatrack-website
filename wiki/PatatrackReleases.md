---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing `CMSSW_10_4_0_pre2_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_4_0_pre2`, using this dedicated release has few advantages:
  - update Eigen and improve compatibility with CUDA:
    - update to the master branch as of Tue Sep 25 20:26:16 2018 +0200,
    - patch Tensorflow accordingly,
    - extend support for matrix inversion and diagonal matrices in CUDA code,
    - fix deprecation warnings for CUDA 10.0;
  - avoid deprecation warnings for CUDA 10.0 in the CUDA API Wrapper;
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
wget http://cmsrep.cern.ch/cmssw/repos/bootstrap.sh -O $VO_CMS_SW_DIR/bootstrap.sh
chmod a+x $VO_CMS_SW_DIR/bootstrap.sh
$VO_CMS_SW_DIR/bootstrap.sh -a slc7_amd64_gcc700 -r cms -path $VO_CMS_SW_DIR setup
```

### Optional: install locally `CMSSW_10_4_0_pre2`
This step is not necessary, but I often found it useful to have a local installation of the base release alongside with the Patatrack one:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 install -y cms+cmssw+CMSSW_10_4_0_pre2
```

### Install `CMSSW_10_4_0_pre2_Patatrack`
Patatrack releases can now be installed by `cmspkg`, using the dedicated repository:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -r cms.patatrack install -y cms+cmssw+CMSSW_10_4_0_pre2_Patatrack
```
