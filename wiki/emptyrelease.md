---
title: "Building a minimal version of the CMSSW framework"
author: "Andrea Bocci"
layout: wiki
resource: true
categories: wiki
activity:  heterogeneouscomputing
<!-- choose one among these possible activities: pixeltracks, heterogeneouscomputing, ml -->
---
### Building a minimal version of the CMSSW framework

#### How to build an "empty" CMSSW working area

By design, a working area for CMSSW inherits libraries, plugins, executavbles, etc. from its "base release".

In order to break this dependency, and rely exclusively on the code that is complied locally, one can start from any (full) release, e.g. `CMSSW_9_4_X_2017-10-29-0000`, and run

~~~
cmsrel CMSSW_9_4_X_2017-10-29-0000
cd CMSSW_9_4_X_2017-10-29-0000
rm -rf .SCRAM/*/Environment
scram setup
scram setup self
cmsenv
~~~
It is advised to include at least the `FWCore/Version` file in any minimal working area
~~~
cd src
git cms-init
git cms-addpkg FWCore/Version
scram b -j4
~~~

#### Minimal EDM framework and dependencies

As of CMSSW 9.4.x, the minimal compile-time and run-time dependencies for `FWCore/Framework` and `cmsRun` are

~~~
git cms-addpkg \
  DataFormats/Common \
  DataFormats/Provenance \
  DataFormats/StdDictionaries \
  DataFormats/TestObjects \
  DataFormats/WrappedStdDictionaries \
  FWCore/Catalog \
  FWCore/Common \
  FWCore/Concurrency \
  FWCore/Framework \
  FWCore/MessageLogger \
  FWCore/MessageService \
  FWCore/Modules \
  FWCore/ParameterSet \
  FWCore/PluginManager \
  FWCore/PythonParameterSet \
  FWCore/SOA \
  FWCore/ServiceRegistry \
  FWCore/Sources \
  FWCore/Utilities \
  FWCore/Version \
  IOPool/TFileAdaptor \
  Utilities/Testing \
  Utilities/Xerces

scram b -j4
~~~

Other useful packages include

~~~
git cms-addpkg \
  FWCore/Integration \
  FWCore/Services \
  IOPool/Common \
  IOPool/Output \
  IOPool/Provenance \
  Utilities/StorageFactory \
  Utilities/XrdAdaptor

scram b -j4
~~~
