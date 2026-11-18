# Simple3DLocate execution comparison

Project-generated curved and stepped CAD models exercise six positive scenes
with 12 expected physical instances per repeat and two sphere negative scenes.
Scene samples differ from training samples and include missing slices, noise,
clutter, coordinate rotations and translations.

Use Simple3DLocate **0.1.0.dev2 or later**.
Both folders contain a baseline recipe, NPZ scenes, ground-truth validation
manifest and complete comparison report. The source generators and all assets
use this repository's MIT license.

```powershell
simple-3d-locate compare cad/part.s3l cad/validation.json --output cad/replay.json --repeats 2
simple-3d-locate compare curved/part.s3l curved/validation.json --output curved/replay.json --repeats 2
python benchmark_execution.py reproduced --repeats 2
```

[summary.json](summary.json) records the measured matrix.
[CAD report](cad/comparison.json) and [curved report](curved/comparison.json)
include every run, settings, poses, stage costs, runtime versions and sample
hashes. Compilation is separate from execution. The score is the mean of
per-scene median execution times, with positive and negative scenes weighted
equally.

The recorded CAD comparison recommends the compact plan with zero missed or
extra detections. The curved family exposes duplicate orientations in one
scene and has no eligible recommendation. Faster plans that fail the supplied
counts/pose tolerances remain excluded. All sphere negatives were rejected.

Results describe these synthetic fused-surface observations on the local test
system. Real sensors, single-view visibility, near-touching parts and symmetry
handling remain qualification work. A GUI comparison without truth poses
checks agreement with its current recipe; it cannot certify that recipe.
