---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing `CMSSW_10_2_4_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_2_4`, using this dedicated release has few advantages:
  - include support for Volta-class GPUs;
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

### Install locally CMSSW_10_2_4
This is necessary to get the latest external packages:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 install -y cms+cmssw+CMSSW_10_2_4
```

### Install the dependencies and `CMSSW_10_2_4_Patatrack`
To install `CMSSW_10_2_4_Patatrack` on top of a recent Patatrack release (e.g. 10.2.3-Patatrack) it is enough to install the updated packages:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -- rpm --prefix=$VO_CMS_SW_DIR -i \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-toolfile+2.1-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cms-git-tools+180901.0-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw-tool-conf+44.0-patatrack2-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw+CMSSW_10_2_4_Patatrack-1-1.slc7_amd64_gcc700.rpm 
```

To install `CMSSW_10_2_4_Patatrack` in a brand new area, it is necessary to install all dependencies:
```
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -- rpm --prefix=$VO_CMS_SW_DIR -i \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda+9.2.148-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-toolfile+2.1-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+llvm-gcc-toolfile+13.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cub+1.8.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-api-wrappers+20180504-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-gdb-wrapper+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+llvm+6.0.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-dxr+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-dxr-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-llvmlite+0.23.2-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-numba+0.37.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+python_tools+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cms-git-tools+180901.0-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw-tool-conf+44.0-patatrack2-1-1.slc7_amd64_gcc700.rpm \
  https://fwyzard.web.cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw+CMSSW_10_2_4_Patatrack-1-1.slc7_amd64_gcc700.rpm 
```


