# Camera and rig calibration revisions

Use **Save calibration profile** to publish a camera model or stereo rig.
Set **Path** to the directory that will hold calibration revisions. Set
**Profile ID** and **Revision** to short lowercase names such as
`station-camera` and `r2`. The block saves a separate JSON snapshot at
`station-camera/r2.json` within that directory. Saving to an existing revision
fails and preserves the file. **Load calibration profile** opens the selected
pair of names and verifies the file checksum. Change **Revision** explicitly
when switching versions. Leaving both fields blank uses the single-file mode.

A camera snapshot uses `camera_model_v1`. For a legacy pinhole lens fit,
connect it through **Camera intrinsic model** to give it a named optical frame.
A stereo snapshot contains the left and right camera models and the rigid
left-to-right transform. Its cameras must have distinct optical frame names.
Connect a loaded camera or rig directly to the appropriate consumer. A loaded
revision carries its frame and numerical values together.

## Record capture metadata

Put **Declare capture calibration revision** near the image source. Enter the
profile ID, revision and camera frame known for that capture. If you have its
recorded SHA-256, enter that too. For a stereo pair use two declarations with
the same profile and revision and the two different optical frames. A project
using recorded images should get those values from its acquisition record.
The block attaches a declaration to the current ticket and preserves an
existing declaration. It does not identify an unlabelled camera from pixels.

A versioned camera accepts a matching declaration and the original full pixel
grid. Crops, resizes and different pyramid levels require their own calibration
workflow. The consumer checks source dimensions, root dimensions and the
pixel-to-root transform. Derived rectified images carry the applied revision
and their new grid. Keep raw and rectified images distinguishable in the graph.

## Apply maps and stereo geometry

**Build lens correction maps** accepts an annotated pinhole model with a
measured RMS. **Save lens correction maps** stores the source revision with the
map pair, and **Load lens correction maps** verifies that metadata and both
planes. **Remap image coordinates** and the packed-map path check the source
capture against the map revision before sampling. The archived file is chosen
explicitly by its path.

**Rectify stereo images** checks the left and right declarations and uses the
transform from the selected rig. Its two output images are tied to the same
rectification record. Disparity and 3D reprojection preserve that link and
reject a mismatched pair or Q matrix. Legacy unversioned files and inputs
continue to use their existing sizing and frame checks.

[Calibration revision projects](<../examples/opencv tools/calibration_revisions/README.md>)
exercise explicit revision selection, retained map metadata and rejection of an
older rig for a newer capture.
