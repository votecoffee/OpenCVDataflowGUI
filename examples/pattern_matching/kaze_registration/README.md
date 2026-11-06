# KAZE registration

Open [kaze_registration.json](kaze_registration.json) with **File → Open…**
and click **Run once**. The graph detects KAZE keypoints in a synthetic
texture and its shifted copy, pairs the descriptors with **Match descriptors
with FLANN**, and estimates a homography. On the bundled fixture, 303 inliers
recover about (9.05, 13.00) pixels of the known (9, 13) pixel shift.

**KAZE keypoints and descriptors** is in **Inspection → Features**.
Choose 64 or 128 floating-point values per descriptor with **Extended**.
Turn on **Upright** to use a fixed orientation for scenes without rotation.
**Threshold**, **Octaves**, **Octave layers**, and **Diffusivity** control
the detector. **Max features** caps the returned keypoints by response.
Connect a mask or region to restrict keypoint centers. The block outputs
descriptors, points, and a preview with keypoint markers.
[OpenCV KAZE reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1KAZE.html)

Connect the two **Features** outputs to the FLANN matcher's **Query** and
**Train** inputs. Use the same **Extended** and **Upright** settings on both
detectors so the descriptors are comparable.

```powershell
opencv-dataflow-run examples/pattern_matching/kaze_registration/kaze_registration.json
```

[All examples](../../README.md)
