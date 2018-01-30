---
title: "Development for Patatrack"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Simple recipe for developing with Patatrack

  - create a local working are as usual
```
# create a local working area
cmsrel CMSSW_10_0_0
cd CMSSW_10_0_0/src/
cmsenv
git cms-init
```
  - and add the patatrack repository
```
# add the patatrack remote repository
git cms-remote add cms-patatrack
git checkout cms-patatrack/CMSSW_10_0_0_Patatrack -b my_development
```
  - write code, compile, debug, commit, and push to your repository
```
scram b
git push my-cmssw HEAD:my_development
```
  - open https://github.com/cms-patatrack/cmssw
  - there should be box with the branch you just created and a green button saying "Compare & pull request":
    ![Compare & pull request](screenshot1.png "Compare & pull request")
  - click on it, and create a pull request as usual
    ![Create a pull request](screenshot2.png "Create a request")

