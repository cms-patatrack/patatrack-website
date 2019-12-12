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
   
