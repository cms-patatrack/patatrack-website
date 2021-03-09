## The HLT menu

The HLT configurations, or "menus", used for data taking can be used offline with minimal changes.
For an up-to-date description of the different configurations, releases and conditions, please check the
[SWGuideGlobalHLT](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGlobalHLT) twiki page.

The configurations used for data taking can be found on [WBM](https://cmswbm.cern.ch/), on [OMS](https://cmsoms.cern.ch/cms/index/index),
or on this [summary page](https://fwyzard.web.cern.ch/fwyzard/hlt/2018/summary.html).

The rest of the instructions will use the HLT menu `/cdaq/physics/Run2018/2e34/v3.6.0/HLT/V4`, from the end of the 2018
pp data taking.

### Extracting the HLT "menu"

The HLT configuration used for data taking can be extracted with the `hltConfigFromDB` command:

```bash
hltConfigFromDB --adg --configName /cdaq/physics/Run2018/2e34/v3.6.0/HLT/V4 > hlt.py
```

In order to re-run the HLT offline, some minor changes are needed:
  - change the process name
  - change the `Source`
  - customise the configuration for the current version of CMSSW

### Change the process name

A simple way to change the process name is to append these lines at the end of the HLT configuration:
```python
process.setName_('TIME')
process.hltHLTriggerJSONMonitoring.triggerResults = cms.InputTag( 'TriggerResults','','TIME' )
process.hltTriggerRatesMonitor.hltResults = cms.untracked.InputTag( 'TriggerResults','','TIME' )
process.hltTriggerBxMonitor.hltResults = cms.untracked.InputTag( 'TriggerResults','','TIME' )
process.hltTriggerObjectTnPMonitor.triggerResults = cms.InputTag( 'TriggerResults','','TIME' )
process.hltTriggerObjectTnPMonitor.triggerEvent = cms.InputTag( 'hltTriggerSummaryAOD','','TIME' )
```

### Change the `Source`

It is possible to use the standard `PoolSource` module to read the events from an EDM root file.
Append these lines at the end of the HLT configuration:

```python
del process.source

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Run2018D/EphemeralHLTPhysics1/RAW/v1/000/323/775/00000/A27DFA33-8FCB-BE42-A2D2-1A396EEE2B6E.root')
)
```

For a more realistic measurement it is also possible to use the DAQ source; see [Using the DAQ Source](DAQSource.md) for this approach.


### Customise the menu for the current CMSSW version

As the configuration of the modules in CMSSW changes, it is sometimes necessary to apply a customisation to the HLT
configuration to allow it to run in the latest release.

Running with the same releases used to take data (in this case `CMSSW_10_1_10`) no customisation should be necessary.

Running with a later relase may requires a customisation; usually, it is enough to append these lines at the end of the HLT configuration:
```python
from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
process = customizeHLTforCMSSW(process)
```


### Configure the number of events to analyse

For a real measurement, order of 10000 events is usually a reasonable value:
```python
# configure the number of events to analyse
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
```

For a quick test it should be enogh to run on 100 events.


### Configure the number of concurrent threads, streams and lumisections

The HLT is usually run with 4 threads and 4 concurrent streams:
```python
# configure the number of concurrent threads, streams and lumisections
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(4),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(False)
)
```

### Configure the process to be verbose

A quick way to check the timing and trigger results is to configure the process to be verbose:
```python
# configure the process to be verbose
process.MessageLogger.categories.append('FastReport')
process.MessageLogger.cerr.FastReport = cms.untracked.PSet( limit = cms.untracked.int32( 10000000 ) )
process.options.wantSummary = cms.untracked.bool(True)
```
The output of the `FastTimerService` is prefixed by `FastReport`; the "trigger report" is prefixed by `TrigReport`;
and the "timing report" embeded in the framework is prefixed by `Timereport`.

## The input data

To perform a realistic measurement of the HLT timing, one should run on an input sample that represents the right
mixture of L1T-selected events; a very good approximation is an "HLTPhysics" sample, which contains "physics" events
selected by the Level 1Trigger without any further HLT selection; while "random" and "calibration" events are missing,
those are a very small fraction (less than 1%) of the HLT input, and can be safely neglected.

The rest of the instructions will use an HLTPhysics input from era Run2018D, fill 7240, run 323775, luminosity
sections 53 and 54 - corresponding to a luminosity of 1.78e34 cm-2s-1 and an average pileup of 49.7:
```
/store/data/Run2018D/EphemeralHLTPhysics1/RAW/v1/000/323/775/00000/A27DFA33-8FCB-BE42-A2D2-1A396EEE2B6E.root
```
