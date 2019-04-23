---
title: "Patatrack Releases"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Installing "Patatrack" CMSSW releases
While it is possible to start from the vanilla `CMSSW_10_5_0` or `CMSSW_10_6_0_pre2` releses, using a dedicated release has few advantages:
  - support for different CUDA versions in different architectures;
  - include the changes from the "Patatrack" development branch.

`CMSSW_10_5_0_Patatrack` is available for
  - `slc7_amd64_gcc700` with CUDA 10.0.130.

`CMSSW_10_6_0_pre2_Patatrack` is available for
  - `slc7_amd64_gcc700` with CUDA 9.2.148, 10.0.130, 10.1.105;
[//]:   - `slc7_aarch64_gcc700` with CUDA 10.0.166.


### Bootstrap a local installation of CMSSW
Choose for `VO_CMS_SW_DIR` a directory to which you have write permissions, and has at least 20 GB of spare disk space.
**Do not** use a directory on EOS.

[//]: # (The following instructions assume the `slc7_amd64_gcc700` architecture; to use a different one simply replace the desired architecture.)

```bash
cd <...>

export LANG=C
export VO_CMS_SW_DIR=$PWD
export SCRAM_ARCH=slc7_amd64_gcc700

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
export SCRAM_ARCH=slc7_amd64_gcc700

$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH upgrade -y
```

## Install `CMSSW_10_5_0_Patatrack`

`CMSSW_10_5_0_Patatrack` and older releases need to be installed in two steps:
first install the corresponding "upstream" release to bring in all the
required extenrals, then install the "patatrack" release itself from the
"cms.patatrack.old" repository.

### Install `CMSSW_10_5_0` and its dependencies
Most of the externals for `CMSSW_10_5_0_Patatrack` need to be installed from the
official repository; the easiest approach is to install them automatically
together with `CMSSW_10_5_0`:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH install -y cms+cmssw+CMSSW_10_5_0
```

### Install `CMSSW_10_5_0_Patatrack`
Patatrack releases can now be installed by `cmspkg`, using the dedicated
repository:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -r cms.patatrack.old install -y cms+cmssw+CMSSW_10_5_0_Patatrack
```

## Install `CMSSW_10_6_0_pre2_Patatrack`
Starting from `CMSSW_10_6_0_pre2_Patatrack`, patatrack releases and all their
dependencies can be installed directly by `cmspkg`, using the dedicated
repository "cms.patatrack".

To install `CMSSW_10_6_0_pre2_Patatrack` built with CUDA 9.2:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -r cms.patatrack install -y cms+cmssw+CMSSW_10_6_0_pre2_Patatrack_CUDA_9_2
```

To install `CMSSW_10_6_0_pre2_Patatrack` built with CUDA 10.0:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -r cms.patatrack install -y cms+cmssw+CMSSW_10_6_0_pre2_Patatrack
```

To install `CMSSW_10_6_0_pre2_Patatrack` built with CUDA 10.1:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -r cms.patatrack install -y cms+cmssw+CMSSW_10_6_0_pre2_Patatrack_CUDA_10_1
```
