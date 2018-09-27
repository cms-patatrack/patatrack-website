---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing `CMSSW_10_2_5_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_2_5`, using this dedicated release has few advantages:
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

### Install locally CMSSW_10_2_5
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 install -y cms+cmssw+CMSSW_10_2_5
```

### Install the dependencies and `CMSSW_10_2_5_Patatrack`
```
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -- rpm --prefix=$VO_CMS_SW_DIR --upgrade --nodeps --replacepkgs \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cms-git-tools+180901.0-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw+CMSSW_10_2_5_Patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw-tool-conf+44.0-patatrack3-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+coral+CORAL_2_3_21-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+coral-tool-conf+2.1-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+distcc-gcc-toolfile+2.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+gcc-toolfile+13.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+icc-gcc-toolfile+3.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+llvm-gcc-toolfile+13.0-patatrack4-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cub+1.8.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda+9.2.148-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-api-wrappers+20180504-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-gdb-wrapper+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+eigen+1f44b667dd9aeeb153284b15fc7fe159d2c09329-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+gbl+V02-01-03-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+llvm+7.0.0-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+lwtnn+2.4-patatrack3-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+professor2+2.2.1-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-dxr+1.0-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-dxr-toolfile+1.0-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-tensorflow+1.6.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-tensorflow-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+python_tools+1.0-patatrack2-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+tensorflow+1.6.0-patatrack-1-1.slc7_amd64_gcc700.rpm
```

### Install CUDA 10.0 (optional)
```
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -- rpm --prefix=$VO_CMS_SW_DIR --upgrade --nodeps --replacepkgs \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-toolfile+2.1-patatrack3-1-1.slc7_amd64_gcc700.rpm \
    https://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda+10.0.130-patatrack-1-1.slc7_amd64_gcc700.rpm
```
