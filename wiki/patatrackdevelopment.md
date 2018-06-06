---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack
The current Patatrack development branch is based on `CMSSW_10_2_X`, and supportes the two architectures `slc7_amd64_gcc630` and `slc7_amd64_gcc700`. Adapt these instructions as needed.

## Installing `CMSSW_10_2_0_pre4_Patatrack2`
While it is possible to start from the vanilla `CMSSW_10_2_0_pre4`, using this dedicated release has few advantages
  - uses the latest version of the relevant externals (llvm/clang, and eigen)
  - includes the recent developments to the `DataFormats`, reducing the number of packages that need to be checked out and built locally

If you are working on **vinavx2** you can skip these steps, as the release is already available.

### Setup a local installation of CMSSW
Choose for `VO_CMS_SW_DIR` a directory to which you have write permissions, and has at least 20 GB of spare disk space.  
**Do not** use a directory on EOS.

```bash
export VO_CMS_SW_DIR=<...>
export SCRAM_ARCH=slc7_amd64_gcc700
export LANG=C
curl http://cmsrep.cern.ch/cmssw/repos/bootstrap.sh -o $VO_CMS_SW_DIR/bootstrap.sh
sed -i -e's/^download_method=.*/download_method=curl/' $VO_CMS_SW_DIR/bootstrap.sh      # prefer curl over wget
chmod a+x $VO_CMS_SW_DIR/bootstrap.sh
$VO_CMS_SW_DIR/bootstrap.sh -a $SCRAM_ARCH -r cms -path $VO_CMS_SW_DIR setup
```

### Install locally CMSSW_10_2_X_2018-06-04-2300
This is necessary to get the latest external packages (llvm and eigen).
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -r cms.week1 install -y cms+cmssw+CMSSW_10_2_X_2018-06-04-2300
```

### Install the dependencies
For `slc7_amd64_gcc630`: 
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc630 -- rpm --prefix=$VO_CMS_SW_DIR -i \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc630/external+dwz+0.12-cms-1-1.slc7_amd64_gcc630.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc630/cms+coral+CORAL_2_3_21-cms2-1-1.slc7_amd64_gcc630.rpm
```

For `slc7_amd64_gcc700`:
```bash
$VO_CMS_SW_DIR/common/cmspkg -a slc7_amd64_gcc700 -- rpm --prefix=$VO_CMS_SW_DIR -i \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/external+dwz+0.12-1-1.slc7_amd64_gcc700.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/slc7_amd64_gcc700/cms+coral+CORAL_2_3_21-cms-1-1.slc7_amd64_gcc700.rpm
```

### Install `CMSSW_10_2_0_pre4_Patatrack2`
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -- rpm --prefix=$VO_CMS_SW_DIR -i \
  http://cern.ch/fwyzard/patatrack/rpms/$SCRAM_ARCH/cms+cmssw-tool-conf+43.0-cms-1-1.$SCRAM_ARCH.rpm \
  http://cern.ch/fwyzard/patatrack/rpms/$SCRAM_ARCH/cms+cmssw+CMSSW_10_2_0_pre4_Patatrack2-1-1.$SCRAM_ARCH.rpm
```

### Optionally, install the debug symbols
```bash
$VO_CMS_SW_DIR/common/cmspkg -a $SCRAM_ARCH -- rpm --prefix=$VO_CMS_SW_DIR -i \
  http://cern.ch/fwyzard/patatrack/rpms/$SCRAM_ARCH/cms+cmssw-debug+CMSSW_10_2_0_pre4_Patatrack2-1-1.$SCRAM_ARCH.rpm
```

## Create a working area for `CMSSW_10_2_0_pre4_Patatrack2`

On **vnavx2** set `VO_CMS_SW_DIR=/data/cmssw`.

### Source the local installation
```bash
export VO_CMS_SW_DIR=<...>
export SCRAM_ARCH=slc7_amd64_gcc700
source $VO_CMS_SW_DIR/cmsset_default.sh
```

### Set up a working area
```bash
scram list CMSSW_10_2_0_pre4
cmsrel CMSSW_10_2_0_pre4_Patatrack2
cd CMSSW_10_2_0_pre4_Patatrack2/src
cmsenv
```

### Set up the `git` repository
```bash
git cms-init --upstream-only || true
# you will see the error
#     fatal: Not a valid object name: 'CMSSW_10_2_0_pre4_Patatrack2'.
# it is expected, just follow the rest of the instructions
git config core.sparsecheckout true
{
  echo "/.gitignore"
  echo "/.clang-tidy"
  echo "/.clang-format"
} > $CMSSW_BASE/src/.git/info/sparse-checkout
git read-tree -mu HEAD

# add the Patatrack remote and branches
git cms-remote add cms-patatrack
git checkout CMSSW_10_2_0_pre4_Patatrack2 -b CMSSW_10_2_X_Patatrack
git branch -u cms-patatrack/CMSSW_10_2_X_Patatrack
git checkout CMSSW_10_2_0_pre4_Patatrack2 -b from-CMSSW_10_2_0_pre4_Patatrack2

# enable the developer's repository
git cms-init
```

Now you should be able to work in the `from-CMSSW_10_2_0_pre4_Patatrack2` branch as you would in a normal CMSSW development area.

### Check out the patatrack development branch
To work on further developments, it is adviced to start from the HEAD of the patatrack branch.

```bash
cmsenv
git checkout cms-patatrack/CMSSW_10_2_X_Patatrack -b my_development
# check out the modified packages and their dependencies
git cms-addpkg $(git diff $CMSSW_VERSION --name-only | cut -d/ -f-2 | sort -u)
git cms-checkdeps -a
```

### Write code, compile, debug, commit, and push to your repository
```bash
...
scram b -j`nproc`
...
git add ...
git commit
git push -u my-cmssw HEAD:my_development
```

## Create a pull request
  - open https://github.com/cms-patatrack/cmssw

  - there should be box with the branch you just created and a green button saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")

  - click on it, and create a pull request as usual:
    ![Create a pull request](screenshot2.png "Create a request")

  - make sure to choose `CMSSW_10_2_X_Patatrack` as the target branch
