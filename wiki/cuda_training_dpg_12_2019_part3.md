---
title: "CUDA Training for Tracker DPG - part 2, using TrackingRecHitSoA"
author: "Vincenzo Innocente"
layout: wiki
resource: true
categories: wiki
---

### Setting up CMSSW
See [part 2](cuda_training_dpg_12_2019_part2.md)

### create and run workflows
cd to your home directory or create a working directory somewhere (not afs....)
   - discorver the available GPU workflows 
   - generate the config files for a "2021 mc"
   - avoid to run it 
   - modify the reconstruction step to read an input file from eos and add a TimingService to print each module that is run
   

### exercise 1 : analyze TrackingRecHit On GPU

let's start from {Matti's documentation](https://github.com/cms-patatrack/cmssw/blob/master/HeterogeneousCore/CUDACore/README.md)

### exercise 2 : convert TrackingRecHit to cilyndical coordinates (as SoA!) and store the result in the event
besides the documentation above I suggest to
   - modify in place https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/HeterogeneousCore/CUDATest/interface/CUDAThing.h
   - make the view by value (see https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h
   - use code from exercise 1, https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/HeterogeneousCore/CUDATest/plugins/TestCUDAProducerGPU.cc and your own exercise from Wednesday
