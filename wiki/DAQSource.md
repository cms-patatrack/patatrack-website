## Using the DAQ Source

To emulate as much as possible the online environment, the HLT input data can be converted into the "fed raw data"
format used by the DAQ, and read using the DAQ source module.

### Preparing the input data

The input data can be converted into the "fed raw data" format using the `EvFDaqDirector` service and the 
`RawStreamFileWriterForBU` output module.

A skeleton configuration is as simple as
```python
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("FED")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Run2018D/EphemeralHLTPhysics1/RAW/v1/000/323/775/00000/A27DFA33-8FCB-BE42-A2D2-1A396EEE2B6E.root')
)

process.load('EventFilter.Utilities.EvFDaqDirector_cfi')
process.EvFDaqDirector.baseDir   = '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1'
process.EvFDaqDirector.buBaseDir = '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1'
process.EvFDaqDirector.runNumber = 323775

process.rawStreamFileWriterForBU = cms.OutputModule("RawStreamFileWriterForBU",
    ProductLabel = cms.untracked.string("rawDataCollector"),
    numEventsPerFile = cms.untracked.uint32(100),
    jsonDefLocation = cms.untracked.string(os.path.expandvars('$CMSSW_RELEASE_BASE/src/EventFilter/Utilities/plugins/budef.jsd')),
    debug = cms.untracked.bool(False),
)

process.endpath = cms.EndPath(process.rawStreamFileWriterForBU)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
```

where `process.EvFDaqDirector.runNumber` should be set to the run number corresponding to the data (which is usually
`1` for Monte Carlo), and `process.EvFDaqDirector.baseDir` and `process.EvFDaqDirector.buBaseDir` should be set to the
parent directory that will contain the output. Before running, create the directory `<buBaseDir>/run<runNumber>/open`,
and create an empty file `fu.lock` inside it.  
For example, if `baseDir` and `buBaseDir` are set to `/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1` and the run
being processed is `323775`:
```
mkdir -p /data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/open
touch /data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/open/fu.lock
```

The same result can be obtained automatically from the job itself:
```python
# create a fake BU lock file
import os.path
workdir = '%s/run%06d/open' % (process.EvFDaqDirector.baseDir.value(), process.EvFDaqDirector.runNumber.value())
if not os.path.isdir(workdir):
  os.makedirs(workdir)
open(workdir + '/fu.lock', 'w').close()
```


### Skimming the RAW data to keep only a subset of the FEDs

To speed up reading the input data, it is possible to skim the RAW data and keep only the FEDs one is interested in;
for example, to keep only the Pixel FEDs, add to the previous configuration:

```python
process.rawDataSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32(
        # SCAL
          735,
        # TCDS FED
         1024,
        # Pixel FEDs, barrel plus
         1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, # 11200,
         1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, # 11212,
         1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, # 11224,
         1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, # 11236,
        # Pixel FEDs, barrel minus
         1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, # 11248,
         1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, # 11260,
         1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, # 11272,
         1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, # 11284,
        # Pixel FEDs, endcap plus
         1296, 1297, 1298, 1299, 1300, 1301, 1302, # 11296,
         1308, 1309, 1310, 1311, 1312, 1313, 1314, # 11308,
        # Pixel FEDs, endcap minus
         1320, 1321, 1322, 1323, 1324, 1325, 1326, # 11320,
         1332, 1333, 1334, 1335, 1336, 1337, 1338, # 11332,
    )
)
process.path = cms.Path(process.rawDataSelector)

process.rawStreamFileWriterForBU.ProductLabel = "rawDataSelector"
```

It is advisable to always keep the SCAL and TCDS FEDs.

Other examples:
```python
        # ECAL FEDs, endcap minus
          601,  602,  603,  604,  605,  606,  607,  608,  609,  661,
        # ECAL FEDs, barrel minus
          610,  611,  612,  613,  614,  615,  616,  617,  618,
          619,  620,  621,  622,  623,  624,  625,  626,  627,  662,
        # ECAL FEDs, barrel plus
          628,  629,  630,  631,  632,  633,  634,  635,  636,
          637,  638,  639,  640,  641,  642,  643,  644,  645,  663,
        # ECAL FEDs, endcap plus
          646,  647,  648,  649,  650,  651,  652,  653,  654,  664,
```


### Configuring the DAQ Source

The data in the "fed raw data" format can be read using the `FedRawDataInputSource` module, adding the two services
it requires (the `FastMonitoringService` and `EvFDaqDirector`), and configuring them with the correct run number:
```python
# read the input in FED RAW data format
process.FastMonitoringService = cms.Service( "FastMonitoringService",
    filePerFwkStream = cms.untracked.bool( False ),
    fastMonIntervals = cms.untracked.uint32( 2 ),
    sleepTime = cms.untracked.int32( 1 )
)

process.EvFDaqDirector = cms.Service( "EvFDaqDirector",
    runNumber = cms.untracked.uint32( 321177 ),

    baseDir = cms.untracked.string( "tmp" ),
    buBaseDir = cms.untracked.string( "tmp" ),

    useFileBroker = cms.untracked.bool( False ),
    fileBrokerKeepAlive = cms.untracked.bool( True ),
    fileBrokerPort = cms.untracked.string( "8080" ),
    fileBrokerUseLocalLock = cms.untracked.bool( True ),
    fuLockPollInterval = cms.untracked.uint32( 2000 ),

    requireTransfersPSet = cms.untracked.bool( False ),
    selectedTransferMode = cms.untracked.string( "" ),
    mergingPset = cms.untracked.string( "" ),

    outputAdler32Recheck = cms.untracked.bool( False ),
)

process.source = cms.Source( "FedRawDataInputSource",
    runNumber = cms.untracked.uint32( 321177 ),
    getLSFromFilename = cms.untracked.bool(True),
    testModeNoBuilderUnit = cms.untracked.bool(False),
    verifyAdler32 = cms.untracked.bool( True ),
    verifyChecksum = cms.untracked.bool( True ),
    useL1EventID = cms.untracked.bool( False ),         # True for MC, False is the default during data taking
    alwaysStartFromfirstLS = cms.untracked.uint32( 0 ),

    eventChunkBlock = cms.untracked.uint32( 240 ),      # 32 is the default during data taking
    eventChunkSize = cms.untracked.uint32( 240),        # 32 is the default during data taking
    maxBufferedFiles = cms.untracked.uint32( 8 ),       #  2 is the default during data taking
    numBuffers = cms.untracked.uint32( 8 ),             #  2 is the default during data taking

    fileListMode = cms.untracked.bool( True ),          # False during data taking
    fileNames = cms.untracked.vstring(*(
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000000.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000001.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000002.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000003.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000004.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000005.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000006.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000007.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000008.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000009.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000010.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000011.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000012.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000013.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000014.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000015.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000016.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0053_index000017.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000000.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000001.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000002.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000003.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000004.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000005.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000006.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000007.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000008.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000009.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000010.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000011.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000012.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000013.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000014.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000015.raw',
        '/data/store/data/Run2018D/EphemeralHLTPhysics1/FED/v1/run323775/run323775_ls0054_index000016.raw',
    ))
)

```
