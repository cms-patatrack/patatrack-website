---
title: "Workflows"
author: "Matti Kortelainen"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## CMSSW Workflows

### Matrix workflows

Below is a table for all the workflows of interest defined in CMSSW. The workflows can be run with `runTheMatrix.py` along
```bash
$ runTheMatrix.py -l 10824.5,10824.52 -j 2
```
See `runTheMatrix.py --help` for more information on the parameters.

These workflows can be used also as a configuration generation, e.g. to run on a different data (some other run, local files, MC with pileup). To only generate configurations, pass `-j 0` argument along
```bash
$ runTheMatrix.py -l 10824.5,10824.52 -j 0
```
and pick the reconstruction (and harvesting) configuration files from the created directories.


#### Data

| Workflow | Description |
| -------- | ----------- |
| 136.8645 | 2018B JetHT data, `pixelTrackingOnly` reconstruction running in CPU |
| 136.86452 | 2018B JetHT data, `pixelTrackingOnly` reconstruction running in GPU |

#### MC

| Workflow | Description |
| -------- | ----------- |
| 10824.5  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction running in CPU |
| 10824.51  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction with Riemann fit running in CPU |
| 10824.52  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction running in GPU |
| 10824.53  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction running in GPU, with Riemann fit run in GPU |

### Customization for profiling

We have a few customization functions to strip down the configuration
for profiling purposes. The customization function can be passed as a
parameter to `cmsDriver.py`, or copy-pasted to an existing
configuration file.

#### Enabling NVTX
In order to enable NVTX events and see CMSSW modules and streams you have to enable the `NVProfilerService` by adding the following to your configuration:
```
process.NVProfilerService = cms.Service("NVProfilerService")
```
#### Profiling workflow

Removes DQM and VALIDATION, replaces the output module with a `GenericConsumer`.

```bash
cmsDriver.py ... --customise RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfiling
```

```python
from RecoTracker.Configuration.customizePixelOnlyForProfiling import customizePixelOnlyForProfiling
process = customizePixelOnlyForProfiling(process)
```

#### Disable the SoA -> legacy conversion

In addition to the profiling workflow above, disables the conversion from SoAs/PODs to legacy EDM formats.

```bash
cmsDriver.py ... --customise RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUWithHostCopy
```

```python
from RecoTracker.Configuration.customizePixelOnlyForProfiling import customizePixelOnlyForProfilingGPUWithHostCopy
process = customizePixelOnlyForProfilingGPUWithHostCopy(process)
```

#### Disable the GPU -> CPU transfers

In addition to disabling the conversion to legacy EDM formats, disables the GPU->CPU transfers altogether.

```bash
cmsDriver.py ... --customise RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly
```

```python
from RecoTracker.Configuration.customizePixelOnlyForProfiling import customizePixelOnlyForProfilingGPUOnly
process = customizePixelOnlyForProfilingGPUOnly(process)
```
