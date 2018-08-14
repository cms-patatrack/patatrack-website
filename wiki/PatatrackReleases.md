---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing `CMSSW_10_2_2_Patatrack`
While it is possible to start from the vanilla `CMSSW_10_2_2`, using this dedicated release has few advantages:
  - update CUDA to version 9.2.148, patch 1;
  - include the Boost.MPI library;
  - include the changes from CMSSW_10_2_2_patch1;
  - include the changes from the CMSSW_10_2_X branch.

If you are working on **vinavx2** you can skip these steps, as the release is already available.

### Bootstrap a local installation of CMSSW
Choose for `VO_CMS_SW_DIR` a directory to which you have write permissions, and has at least 20 GB of spare disk space.  
**Do not** use a directory on EOS.

```bash
export VO_CMS_SW_DIR=<...>
export SCRAM_ARCH=slc7_amd64_gcc700
export LANG=C
wget http://cmsrep.cern.ch/cmssw/repos/bootstrap.sh -O $VO_CMS_SW_DIR/bootstrap.sh
chmod a+x $VO_CMS_SW_DIR/bootstrap.sh
$VO_CMS_SW_DIR/bootstrap.sh -a slc7_amd64_gcc700 -r cms -path $VO_CMS_SW_DIR setup
```

### Install locally CMSSW_10_2_2
This is necessary to get the latest external packages:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 install -y cms+cmssw+CMSSW_10_2_2
```

### Install the dependencies and `CMSSW_10_2_2_Patatrack`
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -- rpm --prefix=$VO_CMS_SW_DIR --nodeps -i \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+boost+1.63.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+boost-toolfile+1.3-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cgal+4.2-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cgal-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+coral-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cub+1.8.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cub-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda+9.2.148-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-api-wrappers+20180504-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-api-wrappers-toolfile+2.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-gdb-wrapper+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-gdb-wrapper-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+cuda-toolfile+2.1-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+dd4hep+v01-08x-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+dd4hep-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+herwigpp+7.1.2-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+herwigpp-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+llvm+6.0.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+lwtnn+2.4-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+lwtnn-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+openmpi+2.1.4-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+openmpi-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-dxr+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-dxr-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-llvmlite+0.23.2-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+py2-numba+0.37.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+python_tools+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+sherpa+2.2.4-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+sherpa-toolfile+2.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+tinyxml+2.5.3-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+tinyxml-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+utm+utm_0.6.7-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+utm-toolfile+1.1-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+yaml-cpp+0.6.2-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+yaml-cpp-toolfile+1.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+llvm-gcc-toolfile+13.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+coral+CORAL_2_3_21-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+coral-tool-conf+2.1-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw-tool-conf+44.0-patatrack-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+cmssw+CMSSW_10_2_2_Patatrack-1-1.slc7_amd64_gcc700.rpm
```
