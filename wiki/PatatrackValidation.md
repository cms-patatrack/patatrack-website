---
title: "Validation of the Patatrack workflows"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  instructions
---

## Running the validation in a local release area

After setting up a Patatrack release and preparing some developing, before submitting a pull request it is good practice to run the
standard validation workflow.

On **vinavx2** this is as easy as running

```bash
cd $CMSSW_BASE
git clone git@github.com:cms-patatrack/patatrack-validation.git
patatrack-validation/validate.local >& log
```

To run on a different machine, see "Running on a different machine" below.

This will create the workflows
  - 10824.5: pixel-only reconstruction, running on the CPU
  - 10824.52: pixel-only reconstruction, running on the GPU
  - 10824.51: pixel-only reconstruction with Riemann fit, running on the CPU
  - 10824.53: pixel-only reconstruction running on the CPU, with Riemann fit running on the GPU

and run them over
  - 200 events from a Zmumu sample without pileup
  - 100 events from a TTbar sample with pileup 50

The "step1" (GEN-SIM) and "step2" (DIGI) jobs are skipped; the RAW data is read directly from the existing relval samples.
The `step3.py` configuration is modified to include the `NVProfilerService` and to print the messages from the `CUDAService`.  
The resulting "step3" (RECO-DQM) job is run within `nvprof`, with the profile summary saved in `step3.profile` and a full
report, suitable for the NVIDIA Visual Profiler, in `step3.nvvp`.  
The "step4" (HARVESTING) job is then run to produce the final DQM results.  
Then, for each sample, the `makeTrackValidationPlots.py` is used to create the standard tracking validation plots in the
`$CMSSW_BASE/plots` directory.

For the CUDA-enabled workflows (10824.52 and 10824.53) the "step3" job is also run multiple times under `cuda-memcheck`, with
different options:
  - `cuda-memcheck --tool initcheck`
  - `cuda-memcheck --tool memcheck --leak-check full --report-api-errors all`
  - `cuda-memcheck --tool synccheck`

The results are saved in `cuda-initcheck.log`, `cuda-memcheck.log` and `cuda-synccheck.log`.

As these workflow include the DQM and Validation steps, they are not suited for profiling and benchmarks.  
For the 10824.5 and 10824.52 workflows a simplified configuration is created: `profile.py`.  
This includes the same customisations for the `NVProfilerService` and `CUDAService`, and is run in the same way under `nvprof`;
the summary is saved in `profile.profile` and the NVVP report in `profile.nvvp`.


## Running the validation of one or more Pull Requests

The `validate` script can be used to run the same tests, with more extensive comparisons across different releases.

By default, it will create a new directory, where it will generate and run
  - the 10824.5 workflow in a "reference" release (e.g. `CMSSW_11_2_0_pre5`);
  - all 10824.5, 10824.52, 10824.51, 10824.53 workflows on a "development" release (e.g. CMSSW_11_2_0_pre5_Patatrack, updated to
  the HEAD of the CMSSW_11_2_X_Patatrack branch).

If one or more [pull reqest](https://github.com/cms-patatrack/cmssw/pulls/) numbers are passed on the command line, an
extra set of workflows is run in a "testing" area, where the pull requests are merged on top of the "development" release.

The DQM plots and their comparison across the dfferent workflows and releases are uploaded to the the ["dev" DQM GUI]([http://dqmgui7.cern.ch:8060/dqm/dev]).

The various logs and the `makeTrackValidationPlots.py` comparisons are uploaded to the user "www" area under `~/www/patatrack/pulls/`.

A report that summarises all the jobs that have been run and their status is saved as `report.md`. If one or more pull requests
were passed on the command line, and the user has a valid OAUTH token sotred in `~/.patatrack-validation.oauth`, the report is
automatically posted on GitHub as a comment to the pull request.


## Customising the samples being used

To change the number of events, set the variables `TTBAR_NUMEVENTS` and `ZMUMU_NUMEVENTS` to the desired values.

To change the dataset being used (for example, to pick a more recent set of relvals), update the dataset names in the variables
`TTBAR` and `ZMUMU`.

The relval samples should automatically be read via xrootd (from EOS if they are avaibale at CERN); they can also be cached on
the local machine for faster access (see "Running on a different machine" below).


## Running on a different machine

The `validate` and `validate.local` scripts are set up for **vinavx2**.  
On a different machine one should:
  - change `TTBAR_CACHE_PATH` and `ZMUMU_CACHE_PATH` to point to local directories;
  - copy to these directories the files mentoned in `TTBAR_CACHE_FILE` and `ZMUMU_CACHE_FILE`, respectively.

For `validate`, one should additionally set `VO_CMS_SW_DIR` to the location of the `cmsset_default.sh` script.
