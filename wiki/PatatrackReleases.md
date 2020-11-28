---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## "Patatrack" CMSSW releases

### `CMSSW_11_1_X_Patatrack` stable releases

The Patatrack stable branch is based on `CMSSW_11_1_X`, and
supports CUDA 11.0.x and GCC 8.3.x.

`CMSSW_11_1_3_Patatrack` is available for the architectures:

  - `slc7_amd64_gcc820` (Intel/AMD, CentOS 7, GCC 8)
  - `slc7_aarch64_gcc820` (ARM, CentOS 7, GCC 8)
  - `slc7_ppc64le_gcc820` (Power,  CentOS 7, GCC 8)
  - `cc8_amd64_gcc8` (Intel/AMD, CentOS 8, GCC 8)


### `CMSSW_11_2_X_Patatrack` developments releases

The Patatrack development branch is based on `CMSSW_11_2_X`, and
supports CUDA 11.1 and GCC 8 and later.
There is a known problem using CUDA 11.1 with GCC 10 in C++17 mode;
NVIDIA is aware and working on a fix.

`CMSSW_11_2_0_pre10_Patatrack` is available for the architectures:

  - `slc7_amd64_gcc820` (Intel/AMD, CentOS 7, GCC 8)
  - `slc7_amd64_gcc900` (Intel/AMD, CentOS 7, GCC 9)
  - `slc7_aarch64_gcc9` (ARM, CentOS 7, GCC 9)
  - `slc7_ppc64le_gcc9` (Power,  CentOS 7, GCC 9)
  - `cc8_amd64_gcc9` (Intel/AMD, CentOS 8, GCC 9)


### Installation area

The `CMSSW_11_1_X_Patatrack` and later releases are available on CVMFS, along
with the standard CMSSW releases:
```bash
export SCRAM_ARCH=slc7_amd64_gcc820
source /cvmfs/cms.cern.ch/cmsset_default.sh
scram list CMSSW_11_1_3_Patatrack

Listing installed projects available for platform >> slc7_amd64_gcc820 <<

--------------------------------------------------------------------------------
| Project Name  | Project Version          | Project Location                  |
--------------------------------------------------------------------------------

  CMSSW           CMSSW_11_1_3_Patatrack
                                         --> /cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_1_3_Patatrack
```
```bash
scram list CMSSW_11_2_0_pre10_Patatrack

Listing installed projects available for platform >> slc7_amd64_gcc820 <<

--------------------------------------------------------------------------------
| Project Name  | Project Version          | Project Location                  |
--------------------------------------------------------------------------------

  CMSSW           CMSSW_11_2_0_pre10_Patatrack
                                         --> /cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_2_0_pre10_Patatrack
```

## Installing "Patatrack" CMSSW releases

Please note: installing the Patatrack releases locally is necessary only for
working on a system without access to CVMFS.

### Bootstrap a local installation of CMSSW
Choose for `VO_CMS_SW_DIR` a directory to which you have write permissions, and
has at least 20 GB of spare disk space. **Do not** use a directory on EOS.

The following instructions assume the `slc7_amd64_gcc820` architecture; to use a
different one simply replace the desired architecture.

```bash
cd <...>

export LANG=C
export VO_CMS_SW_DIR=$PWD
export SCRAM_ARCH=slc7_amd64_gcc820

wget http://cmsrep.cern.ch/cmssw/bootstrap.sh -O $VO_CMS_SW_DIR/bootstrap.sh
chmod a+x $VO_CMS_SW_DIR/bootstrap.sh
$VO_CMS_SW_DIR/bootstrap.sh -a $SCRAM_ARCH -r cms -path $VO_CMS_SW_DIR setup
```

### Updating a local installation of CMSSW
After the initial boostrap, the same directory can be used to install new releases.

It is recommended to update `cmspkg` itself on a regular basis:
```bash
cd <...>

export LANG=C
export VO_CMS_SW_DIR=$PWD
export SCRAM_ARCH=slc7_amd64_gcc820

$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH upgrade -y
```

### `CMSSW_11_1_X_Patatrack` stable branch

The `CMSSW_11_1_X_Patatrack` releases are available on the official repository,
and can be installed directly with:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH install -y cms+cmssw+CMSSW_11_1_3_Patatrack
```

### `CMSSW_11_2_X_Patatrack` development branch

The `CMSSW_11_2_X_Patatrack` releases are available on the official repository,
and can be installed directly with:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH install -y cms+cmssw+CMSSW_11_2_0_pre10_Patatrack
```
