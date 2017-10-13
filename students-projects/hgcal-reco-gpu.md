---
title: "Massively parallel algorithms for the CMS High-Granularity Calorimeter reconstruction"
author: "Felice Pantaleo"
layout: default
markdown: kramdown
resource: true
categories: students-projects
---


### Massively parallel algorithms for the CMS High-Granularity Calorimeter reconstruction

Starting from 2023, during CMS Phase-2, the increased accelerator luminosity
with the consequently increased number of simultaneous proton-proton collisions
(pile-up) will pose significant new challenges to the CMS experiment.  In order
to keep and eventually improve the high performance of the current forward
detectors, the installation of a new End-cap Calorimeter (EC) is foreseen. The
proposed EC can withstand the harsh radiation levels expected and disentangle
the signal event from the very large pileup background thanks to its high
granularity, both in the transverse and longitudinal directions.

The reconstruction of the clusters produced by particles recorded in the
silicon sensors of the EC is one of the most important components in the
interpretation of the detector information of a proton-proton collision. The
complexity of the events at the expected 200PU, though, will make the
reconstruction in the EC especially challenging. Naive reconstruction
algorithms that have to explore many combinations among all possible paths,
will miserably fail since their performance will not scale linearly with the
number of simultaneous proton collisions.

The scaling in CPUs frequency that characterised the past 10 years of computers
architectures has long gone and cannot be blindly relied upon to cope with the
increased complexity of the events that have to be reconstructed.  The
computing paradigm shifted towards many-cores and parallel architectures that
have yet to be fully exploited in the high energy physics software. Graphics
Processing Units (GPUs) are massively parallel architectures that can be
programmed using extensions to the standard C and C++ languages.  The
exploitation of such architectures requires innovative software design
techniques, algorithms and data structures. The development of such tools and
the full utilization of modern architectures and algorithms will make it
possible to cope with the increasing event complexity without giving up neither
in terms of physics reach nor in terms of computing time. The student will
design parallel algorithms for clustering and will also apply Artificial
Intelligence techniques for improving object identification.



#### Project type
PhD Thesis
