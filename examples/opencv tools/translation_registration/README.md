# Translation from matched features

Open [translation_registration.json](translation_registration.json) with
**File → Open…** and click **Run once**. The graph moves the reference
texture 9 pixels right and 13 pixels down. SIFT finds descriptors in both
views. **Estimate image translation** fits the shared displacement, then
**Warp affine** recreates the shifted image using the fit.

Find **Estimate image translation** under **Inspection → Registration**.
Connect a matcher such as **Match SIFT descriptors** or **Match ORB
descriptors**. **RANSAC** uses the configured pixel tolerance.
**LMeDS** uses a median-based cutoff and assumes a majority of correct
pairs. Set refinement iterations to zero to use OpenCV's closed-form
fit over the inliers. [OpenCV translation estimation reference](https://docs.opencv.org/5.0/main_modules/d_projection.html#estimatetranslation2d)

The transform maps query points to train points without rotation or
scaling. The fit reports the candidate shift, native inlier labels,
and residuals in input order. RMS summarizes the final shift's error
over native inliers. The transform is withheld when the fit misses
the configured minimum inlier count or ratio. Wire the original
train image to preview green inliers and red outliers for up to
512 matches.

A connected **Warp affine** or **Warp perspective** waits for its
transform to emit. A rejected fit therefore stops the dependent
warp. With no transform connection, the warp uses its configured matrix.

```powershell
opencv-dataflow-run "examples/opencv tools/translation_registration/translation_registration.json"
```

[All examples](../../README.md)
