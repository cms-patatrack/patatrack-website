---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## "Patatrack" CMSSW releases

### `CMSSW_11_0_X_Patatrack` stable releases

The stable Patatrack release branch is based on `CMSSW_11_0_X`, and
supports both CUDA 10.1 and CUDA 10.2, with GCC 8.3.x.

`CMSSW_11_0_0_Patatrack` is available for the architecture(s)

  - `slc7_amd64_gcc820` and CUDA 10.1 Update 1.

`CMSSW_11_0_0_Patatrack_CUDA_10_2` is available for the architecture(s)

  - `slc7_amd64_gcc820` and CUDA 10.2.

### `CMSSW_11_1_X_Patatrack` development releases

The current Patatrack development branch is based on `CMSSW_11_1_X`, and 
supports CUDA 10.2 with GCC 8.3.x.

`CMSSW_11_1_0_pre4_Patatrack` is available for the architecture(s)

  - `slc7_amd64_gcc820`;
  - `slc7_aarch64_gcc820`.

### Installation area

The `CMSSW_11_1_X_Patatrack` releases are also available on CVMFS.

The `CMSSW_11_0_X_Patatrack` releases are installed locally:

  - on most Patatrack machines they are installed under `/data/cmssw/`;
  - on **cmg-gpu1080** they are installed under `/data/patatrack/cmssw/`.


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


### `CMSSW_11_0_X_Patatrack` stable branch

#### Prerequisite: install `CMSSW_11_0_0` and its dependencies
Most of the externals for `CMSSW_11_0_0_Patatrack` need to be installed from the
official repository; the easiest approach is to install them automatically
together with `CMSSW_11_0_0`:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH install -y cms+cmssw+CMSSW_11_0_0
```

#### Install `CMSSW_11_0_0_Patatrack` for CUDA 10.1
Patatrack releases can be installed with `cmspkg`, using the dedicated repository
"cms.patatrack":
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -r cms.patatrack install -y cms+cmssw+CMSSW_11_0_0_Patatrack
```

#### Install `CMSSW_11_0_0_Patatrack` for CUDA 10.2
Patatrack releases can be installed with `cmspkg`, using the dedicated repository
"cms.patatrack":
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -r cms.patatrack install -y cms+cmssw+CMSSW_11_0_0_Patatrack_CUDA_10_2
```

### `CMSSW_11_1_X_Patatrack` development branch

#### Using `CMSSW_11_1_0_pre4_Patatrack` on CVMFS
The `CMSSW_11_1_X_Patatrack` releases are available on CVMFS, and can be used directly on lxplus.

#### Install `CMSSW_11_1_0_pre4_Patatrack` and its dependencies
The `CMSSW_11_1_X_Patatrack` releases are available on the official repository,
and can be installed directly:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH install -y cms+cmssw+CMSSW_11_1_0_pre4_Patatrack
```
