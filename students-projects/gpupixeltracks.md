---
title: "Pixel Tracks on GPUs at the CMS High Level Trigger"
author: "Felice Pantaleo"
layout: default
markdown: kramdown
resource: true
categories: students-projects
---
### Pixel Tracks on GPUs at the CMS High Level Trigger
During Run 3, the increased luminosity with the consequent increased pile-up will pose significant new challenges for the CMS detector, in particular for the reconstruction of the tracks, that will be heavily affected by the increased occupancy. The quest of significantly reducing the 40 MHz data rate delivered by proton-proton collisions to the detectors, together with the retention of physics signals potentially interesting for searches of new physics phenomena led to the evaluation of modern multi-cores and many-cores architectures for the enhancement of the existing High-Level Trigger (HLT).

The primary goal of the HLT is to apply a specific set of physics selection algorithms on the events read out and accept the events with the most interesting physics content. By its very nature of being a computing system, the HLT relies on technologies that have evolved extremely rapidly but that cannot rely anymore on an exponential growth of frequency guaranteed by the manufacturers. Graphics Processing Units (GPUs) are massively parallel architectures that can be programmed using extensions to the standard C and C++ languages.

In a synchronous system they proved to be highly reliable and showed a deterministic response-time even in the presence of branch divergences. These two features allow them to be perfectly suited to run pattern recognition algorithms on detector data in a trigger environment. These algorithms, like Cellular Automaton, are often parallel by design, and the highly controlled data produced by a particle physics detector reduces the pattern recognition task to its purest form.

From the physics perspective, such an enhancement of the trigger capabilities would allow inclusion of new tracking triggers and the selection of events that are currently not recorded efficiently.


#### Project type
  Master Thesis, PhD Thesis
