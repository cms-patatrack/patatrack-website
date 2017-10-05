---
title: "Riemann Helix Fit for the CMS Track Reconstruction"
author: "Felice Pantaleo"
layout: default
markdown: kramdown
resource: true
categories: students-projects
---
### Riemann Helix Fit for the CMS Track Reconstruction
A track is as a set of measurement points (hits) in the tracker that can be fitted to provide an estimate of a charged particle trajectory in the tracker. The CMS tracker is embedded in a solenoidal magnetic field, hence making the trajectory of a charged particle is a helix. The helix can be described by five parameters, two for the position, two for the direction and one for the curvature at a given reference position along the track. The application of least-squares methods in track fitting has a long history in high-energy physics experiments. Even if the magnetic field is homogeneous, the least-squares problem is non-linear. In order to apply linear least-squares methods for circle fitting, the function that describes the relation between the track parameter vector at one detector layer with the track parameter vector at the following detector layer has to be linearized. This requirement makes least-square method very computational intensive and complex to implement because approximate track parameters need to be known in advance to compute their derivatives at one layer. The Riemann fit is a fast and relatively simple method based on simplifications of the full, non-linear problem. It maps the hits onto a Riemann sphere and fits a plane to the transformed, three-dimensional measurements. Knowledge of the track parameter derivatives is still not required, making the Riemann fit much simpler to implement than the Kalman filter. A generalized Riemann fit efficiently treats situations where multiple scattering cannot be neglected.

#### Project type
  Master Thesis
