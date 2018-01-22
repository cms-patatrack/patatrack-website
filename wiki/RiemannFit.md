---
title: "Riemann Fit"
author: "Felice Pantaleo"
layout: wiki
resource: true
categories: wiki
activity:  hackathon
<!-- choose one among these possible activities: hackathon -->
---

### Setting the environment up

Connect to the machine `felk40.cern.ch`. In case you don't have an account ask Felice.

You will find the starting configuration file for step3 in `/data/run/`.

You will find the dataset in `/data/store/`.

The `CMSSW` version with which you will work is `CMSSW_10_0_0_pre3`.


~~~
cmsrel CMSSW_10_0_0_pre3;
cd CMSSW_10_0_0_pre3/src;
cmsenv;
git cms-merge-topic fwyzard:calabria/GPU_SiPixel_RawToDigi_94X;
scram b -j8
~~~

You're good to go. Have a nice Hackathon.


### Day 1

Starting point:
Riemann Fit paper:
http://iopscience.iop.org/article/10.1088/1742-6596/762/1/012032

Latest source of Riemann Fit code:
https://github.com/cms-tracking/RiemannFit/blob/master/helix_fit.cpp

Code already in CMSSW:
https://github.com/felicepantaleo/cmssw/tree/RiemannFit_9_2_3/RecoPixelVertexing/PixelTrackFitting

First step:
Pull and compile changes from https://github.com/felicepantaleo/cmssw/tree/RiemannFit_9_2_3/RecoPixelVertexing/PixelTrackFitting

Issues fixed:
  * first issue: File in plugin not named correctly
  * Some global variables removed

Next up:
  * add integration tests (CMSSW / FCCSW)


