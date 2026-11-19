# Calibration revision examples

Open one of these projects and click **Run once**. All captures, profiles and
maps are generated from the app's synthetic stereo fixture.

| Project | Expected result |
| --- | --- |
| [stereo_r2.json](stereo_r2.json) | The declared r2 captures match the selected rig. The 0.12 m baseline and 8 px disparity reconstruct a plane about 3 m away. |
| [stereo_mismatch.json](stereo_mismatch.json) | The same r2 captures are presented to r1. Rectification rejects the mismatch before running and no cloud is emitted. |
| [camera_maps.json](camera_maps.json) | Reload a stored r2 pinhole map pair and apply it to the matching left-camera capture. The result is a 240-by-135 image carrying r2 provenance. |

The revision directory has two stereo profiles with different baselines and
two snapshots of the ideal synthetic pinhole camera. **Load calibration
profile** selects one revision by name. The capture declarations are
independent settings and include the digest from the recorded snapshot.
Change only the loader back to r1 in the passing stereo project to see the
intentional rejection.

Use [build_example.py](build_example.py) with `--output` and an empty directory
to reproduce the projects and assets. The fixture uses ideal camera geometry
and zero synthetic reprojection error. Original synthetic assets and projects
are by OpenCVDataflowGUI contributors under [MIT](../../../LICENSE).

[Profile and capture workflow](../../../docs/calibration-revisions.md)
