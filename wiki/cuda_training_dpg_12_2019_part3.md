---
title: "CUDA Training for Tracker DPG - part 3: use TrackingRecHitSoA, creat eyour own `SoA"
author: "Vincenzo Innocente"
layout: wiki
resource: true
categories: wiki
---

### Setting up CMSSW
See [part 2](cuda_training_dpg_12_2019_part2.md)

make sure you can read the file
```bash
edmFileUtil  /store/relval/CMSSW_10_6_1_patch1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_106X_mcRun3_2021_realistic_v3-v1/10000/F43C676F-C0C5-D04B-802E-F5C265084C20.root
```

### create and run workflows
cd to your home directory or create a working directory somewhere (not afs....)
   - discorver the available GPU workflows 
   - generate the config files for a "2021 mc"
   - avoid to run it 
   - modify the reconstruction step to read the input file from eos and add a TimingService to print each module that is run
   - modify it removing any superfluous (output) module so that it loads a minimal number of modules and runs up to the GPU RecHitProducer  (tip: use the AsciiOutputModule)

### exercise 1 : analyze TrackingRecHit On GPU

let's start from [Matti's documentation](https://github.com/cms-patatrack/cmssw/blob/master/HeterogeneousCore/CUDACore/README.md)

if you prefer copy/paste see also
   - https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/HeterogeneousCore/CUDATest/plugins/TestCUDAAnalyzerGPU.cc
   - https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/HeterogeneousCore/CUDATest/plugins/TestCUDAAnalyzerGPUKernel.cu

### exercise 2 : convert TrackingRecHit to cilyndical coordinates (as SoA!) and store the result in the event
besides the documentation above I suggest to
   - modify in place https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/HeterogeneousCore/CUDATest/interface/CUDAThing.h
   - make the view by value (see https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h
   - use code from exercise 1, https://github.com/cms-patatrack/cmssw/blob/CMSSW_11_0_X_Patatrack/HeterogeneousCore/CUDATest/plugins/TestCUDAProducerGPU.cc and your own exercise from Wednesday
   - add your analyzer/producer to the config from the workflow
