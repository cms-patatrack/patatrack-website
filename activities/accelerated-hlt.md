---
title: "Accelerated High-Level Trigger"
author: "Felice Pantaleo"
layout: default
resource: true
categories: activities

---

### {{page.title}}
The reconstruction of the trajectories of charged particles recorded in the silicon pixel and silicon strip detectors is one of the most important components in the interpretation of the detector information of a proton-proton collision. It provides a precise measurement of the momentum of charged particles (muons, electrons and charged hadrons), the identification of interaction points of the proton-proton collision (primary vertex) and decay points of particles with significant lifetimes (secondary vertices).

The increasing complexity of events, due to the growing number of simultaneous proton-proton collisions, will make track reconstruction especially challenging. In fact, algorithms have to explore many combinations before being able to connect traces left by the same particle in the detector.  

The quest of significantly reducing the 40 MHz data rate delivered by proton-proton collisions to the detectors, together with the retention of those events, which are potentially interesting for searches of new physics phenomena, led to the evaluation of modern multi-cores and many-cores computer architectures for the enhancement of the existing computing infrastructure used for the event selection, i.e. the High-Level Trigger (HLT).

The objective of the HLT is to apply a specific set of physics selection algorithms and to accept the events with the most interesting physics content. To cope with the incoming event rate, the online reconstruction of a single event for the HLT has to be done within 220ms on average.
