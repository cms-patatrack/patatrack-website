---
title: "3rd CERN openlab/Intel hands-on workshop on code optimisation"
author: "Felice Pantaleo"
layout: default
markdown: kramdown
resource: true
categories: events
date: "04-05/05/2017"

---

### {{page.title}}



#### Event URL
[Link](https://indico.cern.ch/event/623960/)

#### Description
Andrea Bocci, Vincenzo Innocente, Felice Pantaleo, Marco Rovere attended the Intel optimization workshop: https://indico.cern.ch/event/623960

The aim of this two days event was to profile and optimize the ECAL multifit at HLT, which is one of the main offenders in the HLT CPU time.

#### Getting started



Connect to the machine olhswep23.cern.ch

You will find data in:
~~~
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/0287E51B-569B-E611-BB24-02163E01253A.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/122FB51B-569B-E611-9FB4-02163E0136F7.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/1C550014-569B-E611-B204-FA163E0E72AE.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/3C2EFD13-569B-E611-9948-FA163EFB5A7C.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/58D6EB1A-569B-E611-A8AB-02163E0119F9.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/6C83F018-569B-E611-8F4B-02163E014368.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/9879140F-569B-E611-900E-FA163EEABB8B.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/AE2B091B-569B-E611-859E-02163E014426.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/DAC0C619-569B-E611-B6E8-02163E0121FE.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/E22C611A-569B-E611-8F3E-02163E011AA9.root
/data/store/data/Run2016H/HLTPhysics0/RAW/v1/000/283/876/00000/FAA94613-569B-E611-8164-FA163EB9C37F.root
~~~
{: .language-bash}

To setup a working area and run the full HLT with offline-like ECAL multifit

~~~
bash
cd /data/$USER
export SCRAM_ARCH=slc7_amd64_gcc630
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_9_1_0_pre3
cd CMSSW_9_1_0_pre3/src
scram setup intel-vtune
cmsenv
git cms-init
~~~
{: .language-bash}
Intel VTune uses signal 38 to collect data
~~~
git cms-addpkg FWCore/Utilities

git remote add fwyzard https://github.com/fwyzard/cmssw.git

git fetch fwyzard

git cherry-pick 1127b7a298aa8e97caf8cba07c7fd0d1865b272a

#recompile the ECAL multifit package with debugging symbols
git cms-addpkg RecoLocalCalo/EcalRecProducers
git cms-addpkg RecoLocalCalo/EcalRecAlgos
USER_CXXFLAGS="-g" scram b -j 32

#to run the whole HLT menu (from 2016)
cmsRun /data/fwyzard/2017/hlt_910pre3.py

#to run full ECAL multifit on every event
cmsRun /data/fwyzard/2017/multifit.py
~~~
{: .language-bash}


#### Notes about the code

Producer interface:
~~~
RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitProducer.h
RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitProducer.cc
~~~
{: .language-bash}
Worker algorithm
~~~
RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerMultiFit.h
RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerMultiFit.cc
~~~
{: .language-bash}


#### Profiling with perf

Full command:
~~~
perf record -e branches,branch-misses,bus-cycles,cache-misses,cache-references,cycles,L1-dcache-load-misses -g --call-graph=lbr taskset -c 1 cmsRun multifit.py
~~~
{: .language-bash}
A simpler version is:
~~~
perf record --call-graph=lbr taskset -c 1 cmsRun multifit.py
~~~
{: .language-bash}
To browse the report, use:
~~~
perf report
~~~
{: .language-bash}
Number of iterations before convergence

    692 Iter: 0
2131761 Iter: 1
 502539 Iter: 2
   6633 Iter: 3
   1423 Iter: 4
    418 Iter: 5
    268 Iter: 6
     22 Iter: 7
     15 Iter: 8
     18 Iter: 9
      4 Iter: 10
      1 Iter: 11
      3 Iter: 12
      1 Iter: 14
      1 Iter: 15
      1 Iter: 16
      1 Iter: 17
      1 Iter: 18
      1 Iter: 40
      1 Iter: 45
#### Profiling with Intel VTune
~~~
amplxe-cl -collect advanced-hotspots -strategy=ld-linux.so.2:nt:nt,ld-linux-x86-64.so.2:nt:nt -target-duration-type=long -data-limit=0 -- cmsRun /data/fwyzard/2017/multifit.py
~~~
{: .language-bash}
##### Running the Intel VTune GUI locally
~~~
export INTEL_LICENSE_FILE=28518@lxlicen01.cern.ch,28518@lxlicen02.cern.ch,28518@lxlicen03.cern.ch
export PATH=$PATH:/cvmfs/projects.cern.ch/intelsw/psxe/linux/x86_64/2017/vtune_amplifier_xe_2017.2.0.499904/bin64
scp -r olhswep23:/data/.../r000hs .
amplxe-gui r000hs
~~~
{: .language-bash}

##### Optimization attempts

baseline reference

To measure the impact of the optimisation attempts, we run the ECAL regional reconstruction over 4000 events with 4 "streams" and 4 threads; each test is repeated 10 times, the first results is discarded, and the other 9 are averaged.

The `EcalUncalibRecHitProducer` is configured as used by the HLT, but running the full multifit:
~~~
process.hltEcalUncalibRecHit.algoPSet.activeBXs = ( -5, -4, -3, -2, -1, 0, 1, 2, 3, 4 )
process.hltEcalUncalibRecHit.algoPSet.doPrefitEB = False
process.hltEcalUncalibRecHit.algoPSet.prefitMaxChiSqEB = 15.
process.hltEcalUncalibRecHit.algoPSet.doPrefitEE = False
process.hltEcalUncalibRecHit.algoPSet.prefitMaxChiSqEE = 10.
~~~
{: .language-python}

Using `slc7_amd64_gcc630 / CMSSW_9_1_0_pre3` out of the box, the cpu time spent in the `EcalUncalibRecHitProducer` is:

34.1 ± 0.2 ms



loop over DIGIs inside `EcalUncalibRecHitWorker*::run()`

EcalUncalibRecHitProducer defers the actual reconstruction of the ECAL rechits to a worker object inheriting from EcalUncalibRecHitWorkerBaseClass, calling the run() virtual method for each digi.

We have added a new interface where run() can be called with a full collection of digis, thus reducing the number of virtual calls and the checks done inside each function.
The default implementatio simply calls run() for each individual digi, but an implementation can now specialise it to take advantage of invariants during the loop:
~~~
virtual void run(const edm::Event& evt, const EcalDigiCollection & digis, EcalUncalibratedRecHitCollection & result)
{
    result.reserve(result.size() + digis.size());
    for (auto it = digis.begin(); it != digis.end(); ++it)
        run(evt, it, result);
}
~~~
{: .language-cpp}

The specialisation for `EcalUncalibRecHitWorkerMultiFit` takes advantage of these assumptions and optimisation opportunities

the digis in a collection come from a single subdetector (barrel or endcap), so the check and configuration can be done based on the first element
the pulse vector and covariance matrix can be reused throughput the iteration, saving the call to Zero()  and the following copy
the results are emplaced in the result vector, saving the `push_back()` and the resulting copy
With these changes, the cpu time spent in the `EcalUncalibRecHitProducer` is 33.5 ± 0.2 ms, resulting in a saving of -1.7% .



optimisations inside `PulseChiSqSNNLS`

Changing a division to a multiplication on line 207:


~~~
diff --git a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
index 98e7252..bbcc7af 100644
--- a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
+++ b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
@@ -204,7 +204,7 @@

bool PulseChiSqSNNLS::DoFit(const SampleVector &samples, const SampleMatrix &sam
   double sigmaplus = std::abs(xplus100-x0)/sqrt(chisqplus100-chisq0);

   //if amplitude is sufficiently far from the boundary, compute also the lower uncertainty and average them
-  if ( (x0/sigmaplus) > 0.5 ) {
+  if (x0 > 0.5 * sigmaplus) {
     for (unsigned int ipulse=0; ipulse<_npulsetot; ++ipulse) {
       if (_bxs.coeff(ipulse)==0) {
         ipulseintime = ipulse
~~~
{: .language-cpp}

results in no measurable changes; the cpu time is 34.1 ± 0.1 ms, consistent with the baseline measurement.



Rewriting the check for positive values without computing the minimum value on lines 379-381 (and adjusting the code on line 398):
~~~
diff --git a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
index 98e7252..ef73b51 100644
--- a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
+++ b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
@@ -376,9 +376,11 @@ bool PulseChiSqSNNLS::NNLS() {
       eigen_solve_submatrix(aTamat,aTbvec,ampvecpermtest,_nP);

       //check solution
-      auto ampvecpermhead = ampvecpermtest.head(_nP);
-      if ( ampvecpermhead.minCoeff()>0. ) {
-        _ampvec.head(_nP) = ampvecpermhead.head(_nP);
+      bool positive = true;
+      for (unsigned int i = 0; i < _nP; ++i)
+        positive &= (ampvecpermtest(i) > 0);
+      if (positive) {
+        _ampvec.head(_nP) = ampvecpermtest.head(_nP);
         break;
       }      

@@ -398,7 +400,7 @@ bool PulseChiSqSNNLS::NNLS() {
         }
       }

-      _ampvec.head(_nP) += minratio*(ampvecpermhead - _ampvec.head(_nP));
+      _ampvec.head(_nP) += minratio*(ampvecpermtest.head(_nP)- _ampvec.head(_nP));

       //avoid numerical problems with later ==0. check
       _ampvec.coeffRef(minratioidx) = 0.;
~~~
{: .language-cpp}

results in a small improvement; the cpu time is 33.8 ± 0.1 ms, resulting in a saving of -0.7% .



Changing the check in the inner loop to increse threshold by a factor 10 every 50 iterations, to a factor 2 every 16 iterations
~~~
diff --git a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
index 98e7252..d283b47 100644
--- a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
+++ b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
@@ -410,8 +410,8 @@ bool PulseChiSqSNNLS::NNLS() {

     //adaptive convergence threshold to avoid infinite loops but still
     //ensure best value is used
-    if (iter%50==0) {
-      threshold *= 10.;
+    if (iter % 16 == 0) {
+      threshold *= 2;
     }
   }
~~~
{: .language-cpp}

results in a small improvement; the cpu time is 33.7 ± 0.1 ms, resulting in a saving of -1.0% .
Note that in our tests, the loop never used more than 11 iterations anyway.



From Vincenzo: using laziproduct() and noalias() in the NNLS:
~~~
diff --git a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
index 98e7252..0bd769c 100644
--- a/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
+++ b/RecoLocalCalo/EcalRecAlgos/src/PulseChiSqSNNLS.cc
@@ -328,9 +328,8 @@ bool PulseChiSqSNNLS::NNLS() {
   constexpr unsigned int nsamples = SampleVector::RowsAtCompileTime;

   invcovp = _covdecomp.matrixL().solve(_pulsemat);
-  aTamat = invcovp.transpose()*invcovp; //.triangularView<Eigen::Lower>()
-  //aTamat = aTamat.selfadjointView<Eigen::Lower>();  
-  aTbvec = invcovp.transpose()*_covdecomp.matrixL().solve(_sampvec);  
+  aTamat.noalias() = invcovp.transpose().lazyProduct(invcovp);
+  aTbvec.noalias() = invcovp.transpose().lazyProduct(_covdecomp.matrixL().solve(_sampvec));

   int iter = 0;
   Index idxwmax = 0;
~~~
{: .language-cpp}
results in the single biggest improvement; the cpu time is 31.3 ± 0.1 ms, resulting in a saving of -8.2% .


| changes          | cpu time         | performance gain     |
|  --------------------   |   --------------  |   --------------   |
|   CMSSW_9_1_0_pre3          |    34.1 ± 0.2 ms | reference   |
| loop over DIGIs | 33.5 ± 0.2 ms | -1.7% |
| division to multiplication |	34.1 ± 0.1 ms|	none|
|positive check| 33.8 ± 0.1 ms | -0.7%|
|inner loop threshold| 33.7 ± 0.1 ms | -1.0%|
|`laziproduct()` and `noalias()`|	31.3 ± 0.1 ms	|-8.2%|


##### Compilation flags

The compilation flags used for `RecoLocalCalo/EcalRecAlgos` and `RecoLocalCalo/EcalRecProducers/plugins` are respectively `-Ofast` and `-O2` . We have tried changing both to `-O3`, enabling AVX2, and FMA:

| changes          | cpu time         | performance gain     |
|  --------------------   |   --------------  |   --------------   |
|  -Ofast / -O2         |    34.1 ± 0.2 ms| reference   |
|-O3|33.5 ± 0.2 ms|-1.7%|
|-Ofast / -O2 -mavx2 | 33.5 ± 0.2 ms | -1.7%|
|-Ofast / -O2 -mavx2 -mfma | 32.6 ± 0.1 ms | -4.3% |



### Summary

Including all changes except for -mavx2 and -mfm2, the cpu time is **29.7 ± 0.1 ms**, resulting in a total saving of **-12.8%** .

Including also -mavx2 and -mfm2, the cpu time is **29.1 ± 0.2 ms**, resulting in a total saving of **-14.7%** .
