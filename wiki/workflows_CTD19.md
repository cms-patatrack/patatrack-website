# Workflows for Connecting The Dots 2019

## CPU workflow

The `cmsDriver.py` command are adapted from `runTheMatrix.py -n -e -l 10824.5`

### step3 (reconstruction, validation, DQM)
```
cmsDriver.py step3 \
  --era Run2_2018 \
  --conditions 102X_upgrade2018_design_v9 \
  --geometry DB:Extended \
  -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM \
  -n 100 \
  --filein file:step2.root \
  --fileout file:step3.root \
  --eventcontent RECOSIM,DQM \
  --datatier GEN-SIM-RECO,DQMIO \
  --runUnscheduled \
  --nThreads 8 \
  --no_exec \
  --python_filename=step3.py
```

### profiling (reconstruction only)
```
cmsDriver.py profile \
  --era Run2_2018 \
  --conditions 102X_upgrade2018_design_v9 \
  --geometry DB:Extended \
  -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly \
  --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfiling \
  -n 100 \
  --filein file:step2.root \
  --no_output \
  --runUnscheduled \
  --nThreads 8 \
  --no_exec \
  --python_filename=profile.py
```

## GPU workflow, with Riemann fit

The `cmsDriver.py` command are adapted from `runTheMatrix.py -n -e -l 10824.52`

### step3 (reconstruction, validation, DQM)
```
cmsDriver.py step3 \
  --era Run2_2018 \
  --conditions 102X_upgrade2018_design_v9 \
  --geometry DB:Extended \
  -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM \
  --procModifiers gpu \
  --customise_commands='process.pixelTracksHitQuadruplets.useRiemannFit = True' \
  -n 100 \
  --filein file:step2.root \
  --fileout file:step3_Riemann.root \
  --eventcontent RECOSIM,DQM \
  --datatier GEN-SIM-RECO,DQMIO \
  --runUnscheduled \
  --nThreads 8 \
  --no_exec \
  --python_filename=step3_Riemann.py
```

### profiling (reconstruction only)
```
cmsDriver.py profile \
  --era Run2_2018 \
  --conditions 102X_upgrade2018_design_v9 \
  --geometry DB:Extended \
  -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly \
  --procModifiers gpu \
  --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfilingDisableConversion \
  --customise_commands='process.pixelTracksHitQuadruplets.useRiemannFit = True' \
  -n 100 \
  --filein file:step2.root \
  --no_output \
  --runUnscheduled \
  --nThreads 8 \
  --no_exec \
  --python_filename=profile_Riemann.py
```

## GPU workflow, with broken line fit

The `cmsDriver.py` command are adapted from `runTheMatrix.py -n -e -l 10824.52`

### step3 (reconstruction, validation, DQM)
```
cmsDriver.py step3 \
  --era Run2_2018 \
  --conditions 102X_upgrade2018_design_v9 \
  --geometry DB:Extended \
  -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM \
  --procModifiers gpu \
  --customise_commands='process.pixelTracksHitQuadruplets.useRiemannFit = False' \
  -n 100 \
  --filein file:step2.root \
  --fileout file:step3_BrokenLine.root \
  --eventcontent RECOSIM,DQM \
  --datatier GEN-SIM-RECO,DQMIO \
  --runUnscheduled \
  --nThreads 8 \
  --no_exec \
  --python_filename=step3_BrokenLine.py
```

### profiling (reconstruction only)
```
cmsDriver.py profile \
  --era Run2_2018 \
  --conditions 102X_upgrade2018_design_v9 \
  --geometry DB:Extended \
  -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly \
  --procModifiers gpu \
  --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfilingDisableConversion \
  --customise_commands='process.pixelTracksHitQuadruplets.useRiemannFit = False' \
  -n 100 \
  --filein file:step2.root \
  --no_output \
  --runUnscheduled \
  --nThreads 8 \
  --no_exec \
  --python_filename=profile_BrokenLine.py
```


