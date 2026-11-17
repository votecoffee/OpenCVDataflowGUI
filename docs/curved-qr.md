# Curved QR labels

Use **Read QR codes** (`inspect.read_qr`) with **Backend: standard** and
**Mode: curved**. It calls OpenCV's `QRCodeDetector.detectAndDecodeCurved` once
per activation. Eps X, Eps Y and alignment-marker controls apply normally.
ArUco/curved is invalid. There is no automatic flat-decoder retry or selection
of another mode when the chosen decoder misses a symbol.

Open the [seven original projects](<../examples/opencv tools/curved_qr/README.md>)
to compare gentle curvature, a tilted cylindrical wrap, a quarter-turn, a larger
symbol, a selected region and two declared failures. Inspect **Overlay** for the
decoded text, native outline and image-plane axes. **Rectified** shows sampled
binary modules, while the hidden-by-default **Info** output describes the read.
The filmstrip receives the overlay, including its graphics.

## Geometry and downstream use

`codes` retain native source-image coordinates and their mean center. Records
identify `backend`, `mode` and `corner_model: native_detection_quadrilateral`.
For a curved label this quadrilateral is an approximation, rather than the
actual surface boundary. In the tilted fixtures the native fourth corner is
over 40 pixels away from the analytically projected surface corner, even though
the text and module grid decode. The other three corners are within three
pixels of their independent finder-associated surface references. These bounds
apply to these fixtures; they are not general localization accuracy limits.

The overlay/graphics preserve that native quadrilateral. Its image-plane axes
show orientation, with no calibrated surface pose. Do not use its fourth corner
as a precise inspection ROI, physical edge, or robot pick coordinate without
separate localization/calibration evidence. Native output includes no sampled
curved outline and no inverse surface map.

`rectified` is a module-resolution observation on a fresh spatial root per read.
It is independent of the camera grid, including when two labels have identical
dimensions. Its `qr_observation` payload records text, backend/mode, source
quadrilateral, source spatial/pyramid context and `mapping_to_source: null`.
Acquisition ticket and recipe revision survive the change of grid. The source
pyramid stays nested as provenance; it is not applied as a module-image scale.
There is no supplied homography from modules back to the curved source.

The native sampled module image may contain incorrect module values even when
QR error correction recovers the correct text. The quarter-turn fixture allows
one differing module; the other passing fixtures exactly match their original
module matrices on the qualified build. The module image is not a certified,
error-corrected re-encoding or a print-quality grade. Use the decoded text for
payload comparison and retain the observed pixels separately when reviewing
image quality.

`info` uses `model: qr_read_v1`. It reports backend, mode, decoded count, corner
model and rectification availability/geometry/value semantics. Start/finish
event data includes `qr_backend` and `qr_mode`. A valid miss returns empty codes,
zero count, no rectified emission and the usual minimum-read judgment/reason.
It does not reuse a previous rectified image. Empty selections skip native
decoding while still reporting the selected mode. Exceptions use ordinary block
containment and a later valid invocation can recover.

## Images and qualification

Selection and type contracts are shared with the [QR backends](qr-backends.md):
uint8 gray or tagged RGB/BGR(A), masks/regions whiten excluded pixels without
cropping or shifting the source grid, and full quiet zones should be retained.
Use explicit upstream depth conversion for other types. Disable **Draw overlay**
and result graphics independently when they are not needed; structured results
and rectified observations remain available.

Original MIT fixtures use a forward pinhole/cylinder projection for independent
surface references and inverse ray/cylinder intersection for raster generation.
Native QR encoding supplies the original, stored module matrices. Labels have
four quiet-zone modules and ten source pixels per module before projection.
These are tool examples, not production photos or calibrated physical captures.
Their full geometry, hashes and expected reads are in `truth.json`; they do not
derive corner references or expected module values from the decoder.

On Python 3.14.3/OpenCV 5.0.0 contrib 5.0.0.93, curved mode reads five cases;
flat mode reads only the mild one. Severe curvature and a 27-degree view fail
both modes and remain explicit inspection failures. This is fixture evidence,
not a general robustness ranking or a promise to read arbitrary surfaces.
Native references: [curved decoder implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/src/qrcode.cpp)
and [OpenCV's separate curved-reader regressions](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/test/test_qrcode.cpp).
The projects include no copied upstream image or source assets.
