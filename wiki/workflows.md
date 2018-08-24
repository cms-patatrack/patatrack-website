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
$ runTheMatrix.py -l 10824.5,10824.8 -j 2
```
See `runTheMatrix.py --help` for more information on the parameters.

These workflows can be used also as a configuration generation, e.g. to run on a different data (some other run, local files, MC with pileup). To only generate configurations, pass `-j 0` argument along
```bash
$ runTheMatrix.py -l 10824.5,10824.8 -j 0
```
and pick the reconstruction (and harvesting) configuration files from the created directories.


#### Data

| Workflow | Description |
| -------- | ----------- |
| 136.8645 | 2018B JetHT data, `pixelTrackingOnly` reconstruction running in CPU |
| 136.8648 | 2018B JetHT data, `pixelTrackingOnly` reconstruction running in GPU |

#### MC

| Workflow | Description |
| -------- | ----------- |
| 10824.5  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction running in CPU |
| 10824.7  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction with Riemann fit running in CPU |
| 10824.8  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction running in CPU |
| 10824.9  | 2018 TTbar noPU, `pixelTrackingOnly` reconstruction running in CPU, with Riemann fit run in GPU |

### Customization for profiling

We have a few customization functions to strip down the configuration
for profiling purposes. The customization function can be passed as a
parameter to `cmsDriver.py`, or copy-pasted to an existing
configuration file.

#### Profiling workflow

Removes DQM and VALIDATION, replaces output module with a dummy `AsciiOutputModule`.

```bash
cmsDriver.py ... --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfiling
```

```python
from RecoPixelVertexing.Configuration.customizePixelTracksForProfiling import customizePixelTracksForProfiling
process = customizePixelTracksForProfiling(process)
```

#### Disable SOA->legacy conversion

In addition to the profiling workflow above disables the "conversion to legacy formats".

```bash
cmsDriver.py ... --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfilingDisableConversion
```

```python
from RecoPixelVertexing.Configuration.customizePixelTracksForProfiling import customizePixelTracksForProfilingDisableConversion
process = customizePixelTracksForProfilingDisableConversion(process)
```

#### Disable GPU->CPU transfers

In addition to disabling the "conversion to legacy" above disables the GPU->CPU transfers altogether.

```bash
cmsDriver.py ... --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfilingDisableTransfer
```

```python
from RecoPixelVertexing.Configuration.customizePixelTracksForProfiling import customizePixelTracksForProfilingDisableTransfer
process = customizePixelTracksForProfilingDisableTransfer(process)
```
