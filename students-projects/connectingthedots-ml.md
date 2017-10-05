---
title: "Connecting the dots with Machine Learning"
author: "Felice Pantaleo"
layout: default
markdown: kramdown
resource: true
categories: students-projects
---

### Connecting the dots with Machine Learning
During Run 4, the increased luminosity with the consequent increased pile-up will pose significant new challenges for CMS detector, in particular for the reconstruction of tracks in the silicon tracker and inside showers in the new High Granularity Calorimeter. When a charged particle produced after the collision flies through the detector, it leaves traces (hits).
Tracking is the art of connecting the correct traces creating trajectories. Timing can easily explode in reconstructing “which dot belongs to which particle” due to combinatorics and it will be heavily affected by the increased track density expected in the next decade. It is of paramount importance to choose the correct paths to follow and the correct hits to add to the path.
Start of many track finding algorithms is the building of track seeds: groups of 2 or 3 measurements that are compatible with each other and  with a crude track hypothesis. Compatibility between two hits can be also based on the hit shape.
These seeds are used to build roads to find track candidates through a progressive filter:
roads are built from track seeds and define a search window
following the road direction to find hits that are compatible with the track
a found hit used to update the track to follow to the next measurement layer needs a mechanism to update a track hypothesis
multiple hypothesis can be tested for one layer
only one track hypothesis is followed further needs a measure which candidate is better
Machine learning techniques can come in aid to reduce the combinatorics and find more stringent compatibility requirements for mitigating the combinatorial explosion.
#### Project type
  Master Thesis
