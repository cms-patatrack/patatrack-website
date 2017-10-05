---
title: "Smart data structures in CUDA"
author: "Felice Pantaleo"
layout: default
markdown: kramdown
resource: true
categories: students-projects
---
### Smart data structures in CUDA

  The entire CMS event reconstruction is written in C++, configured in python enabling the use of dynamic data structures with high granularity. This feature is very useful because typical High-Energy Physics algorithms are fed with variable size inputs and produce outputs, whose size is many times impossible to estimate or generalize. The CMS software framework is going through a transition aimed at transforming it into a heterogeneous software framework, in order to profit of GPUs higher throughput and energy efficiency.

  CUDA language is used to program NVIDIA GPUs and it is not dynamic-data-structures-friendly, since it was designed for working with data structures of predictable size. Furthermore, allocating memory on a GPU has a fixed cost which is orders of magnitude higher than doing that on the host.

#### Project type
  GSoC, Master Thesis
