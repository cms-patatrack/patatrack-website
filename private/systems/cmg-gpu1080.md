---
title: "cmg-gpu1080"
author: "Felice Pantaleo"
layout: default
---
# cmg-gpu1080

### System information:
[Topology of the machine](https://fpantale.web.cern.ch/fpantale/out.pdf)
### OS and drivers available:
CentOS 7, CUDA 9.2
### Getting access to the machine

In order to get access to the machine you should send a request to subscribe to the CERN e-group: cms-gpu
You should also send an email to [Felice Pantaleo](mailto:felice.pantaleo@cern.ch) motivating the reason for the requested access.

#### Usage Policy
Normally, no more than 1 GPU per users should be used. A booking system is being set up.

For booking the machine for exclusive use (no more than 24h consecutively): send an email to [Felice Pantaleo](mailto:felice.pantaleo@cern.ch)

#### Usage for ML studies
If you need to use the machine for training DNNs you could accidentally occupy all the GPUs, making them unavailable for other users.

For this reason you're kindly asked to use

`import setGPU`

before any import that will use a GPU (e.g. tensorflow). This will assign to you the least loaded GPU on the system.

It is strictly forbidden to use GPUs from within your notebook. Please export your notebook to a python program and execute it. The access to the machine will be revoked when failing to comply to this rule.
