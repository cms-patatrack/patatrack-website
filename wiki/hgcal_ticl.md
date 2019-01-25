---
title: "TICL for HGCAL"
author: "Marco Rovere"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## TICL
TICL is the iterative framework that is actively developed to reconstruct
objects within the HGCAL detector. You can find more details about the general
ideas behind it, together with some more technical details about its current
implementation at the following links:

- [Link](https://indico.cern.ch/event/758005/contributions/3143077/attachments/1718869/2773968/20180919_Felice_HGC_IterativeClustering.pdf)
- [Link](https://indico.cern.ch/event/776045/contributions/3225889/attachments/1765337/2865889/20181204_HGCAL_DPG_CMSWEEK_MR.pdf)
- [Link](https://indico.cern.ch/event/783961/contributions/3262281/attachments/1783164/2902090/20190123_HGCAL_AR_MR.pdf)

## CMSSW Workflows

The currently most advanced branch in GitHub that contains a prototype of the
TICL implementation is here:
[link](https://github.com/rovere/cmssw/tree/TICL). It is currently based upon
`CMSSW_10_4_0_pre1`. As soon as `CMSSW_10_5_0_pre1` will be released, it
will be rebased on top of that one.

The current procedure to try it out locally is the following:

```bash
cmsenv CMSSW_10_4_0_pre1
cd CMSSW_10_4_0_pre1/src
cmsenv
git cms-init
git cms-merge-topic rovere:TICL
scram b -j
```

Due to a change a `DataFormats` package, this will download quite a number of
packages and the compilation time could be quite long, depending on
the machine you are using to develop.

### RunTheMatrix workflows useful for HGCAL development

Below is a table for some of the workflows of interest defined in CMSSW that use
**D28** geometry. The workflows can be run with `runTheMatrix.py` along

```bash
$ runTheMatrix.py -l 10824.5,10824.8 -j 2
```
See `runTheMatrix.py --help` for more information on the parameters.

These workflows can be used also as a configuration generation, e.g. to run on a different data (some other run, local files, MC with pileup). To only generate configurations, pass `-j 0` argument along
```bash
$ runTheMatrix.py -l 10824.5,10824.8 -j 0
```
and pick the digitization and reconstruction configuration files from the created directories.

####  Monte Carlo

| Workflow | Description |
| -------- | ----------- |
| 24088.0  | SinglePiPt25Eta1p7_2p7_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24089.0  | SingleMuPt15Eta1p7_2p7_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24090.0  | SingleGammaPt25Eta1p7_2p7_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24091.0  | SingleElectronPt15Eta1p7_2p7_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24000.0  | FourMuPt_1_200_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24001.0  | SingleElectronPt10_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24002.0  | SingleElectronPt35_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24003.0  | SingleElectronPt1000_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24004.0  | SingleGammaPt10_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24005.0  | SingleGammaPt35_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24006.0  | SingleMuPt1_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24007.0  | SingleMuPt10_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24008.0  | SingleMuPt100_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24009.0  | SingleMuPt1000_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24010.0  | FourMuExtendedPt_1_200_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24011.0  | TenMuExtendedE_0_200_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24012.0  | DoubleElectronPt10Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24013.0  | DoubleElectronPt35Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24014.0  | DoubleElectronPt1000Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24015.0  | DoubleGammaPt10Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24016.0  | DoubleGammaPt35Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24017.0  | DoubleMuPt1Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24018.0  | DoubleMuPt10Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24019.0  | DoubleMuPt100Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24020.0  | DoubleMuPt1000Extended_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24021.0  | TenMuE_0_200_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24022.0  | SinglePiE50HCAL_pythia8_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24023.0  | MinBias_13TeV_pythia8_TuneCUETP8M1_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24024.0  | TTbar_13TeV_TuneCUETP8M1_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24026.0  | QCD_Pt_600_800_13TeV_TuneCUETP8M1_2023D28_GenSimHLBeamSpotFull+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24027.0  | Wjet_Pt_80_120_14TeV_TuneCUETP8M1_2023D28_GenSimHLBeamSpotFull14+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24028.0  | Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_2023D28_GenSimHLBeamSpotFull14+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24030.0  | QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_2023D28_GenSimHLBeamSpotFull14+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |
| 24031.0  | QCD_Pt_80_120_14TeV_TuneCUETP8M1_2023D28_GenSimHLBeamSpotFull14+DigiFullTrigger_2023D28+RecoFullGlobal_2023D28+HARVESTFullGlobal_2023D28 |


### How to Activate TICL in your workflow

In the `RecoHGCal/TICL/test` directory there are examples of a test files that
could be used as a skeleton to run TICL alone or together with the **ordinary
reconstruction** and to save the TICL products in the output `ROOT` file.

It will be generally easier to start from the configuration files obtained
starting from one the `RunTheMatrix` workflows listed above and then edit the
`step3` file and add the following code snippet:

```python
+#from ticl_iterations import TICL_iterations
+from ticl_iterations import TICL_iterations_withReco
process = TICL_iterations(process)
```

just before the `process.schedule` definition. Remember also to add

```python
process.TICL
```

to the `process.schedule` definition. Look at the included examples to have a
better understanding of the required changes.

You have to comment/uncomment the proper include customization function from the
`ticl_iterations` package according to your needs. In particular use
`TICL_iterations` to have a fully stand-alone reconstruction based on TICL,
without running anything else from the **ordinary reconstruction**.  This will
define and load also the digitization and clusterisation modules for you,
automatically. Use instead the `TICL_iterations_withReco` to **add the TICL
iterations** to a configuration file that already contains the **ordinary
reconstruction sequence**. This is the preferred way if you edit the
configurations as produced by `RunTheMatrix`.

In all cases the `outputCommands` are extended in order to persiste the
products produced by TICL.

