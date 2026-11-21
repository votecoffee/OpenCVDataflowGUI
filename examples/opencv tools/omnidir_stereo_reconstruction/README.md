# Reconstruct metric omnidirectional stereo

Open the [example project](omnidir_stereo_reconstruction.json) and run once.
Two calibrated Mei cameras observe a **1.55 m** foreground plane and a
**2.44 m** background plane. The project reconstructs the scene on both
perspective and longitude/latitude grids. Select either block to inspect
its rectified views, disparity, depth and colored cloud.

| Projection | Valid disparity pixels | Sampled cloud points | Background XYZ error at the 90th percentile |
| --- | ---: | ---: | ---: |
| Perspective | 78,892 | 8,808 | 35 mm |
| Longitude/latitude | 78,686 | 8,774 | 82 mm |

The two branches use the same input pair and an excluded patch on the raw
right view. The block checks remapped support in both matching footprints.
The disparity output preserves native pixel units and uses NaN where
support or depth checks fail. Cloud XYZ and Depth use meters in the
rectified left frame. Depth is measured along that frame's Z axis.
Cloud colors come from the rectified left pixel associated with each sample.

The block uses native stereo rectification and matching. It triangulates
supported angular rays and retains valid disparities at or below the
native compact cloud's 15 px cutoff. The [reference geometry](assets/truth.json)
records the two planes, camera models and relative pose. The original
synthetic captures are by OpenCVDataflowGUI contributors under the
project's MIT license. Run the builder with an empty output directory
to regenerate the example.
[OpenCV omnidirectional stereo implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ccalib/src/omnidir.cpp#L1271)
