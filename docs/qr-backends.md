# QR reader backends

**Read QR codes** (`inspect.read_qr`) offers a **Backend** dropdown: `standard`
(`QRCodeDetector`) or `aruco` (`QRCodeDetectorAruco`). Existing saved programs
keep the standard backend. Both produce the same `codes`, `count`, `overlay`
and optional `rectified` outputs; downstream text checks need no changes.
There is no automatic fallback or combination of the backends' detections.

Find the reader under **OpenCV → Object Detection (objdetect) → Code reading**.
Open one of the [six comparison projects](<../examples/opencv tools/qr_backends/README.md>)
to inspect both branches on the same filename-ticketed image. No plugin, model
download or optional barcode dependency is required.

## Modes and controls

| Controls | Applies to | Meaning |
| --- | --- | --- |
| Mode `single` / `multi` | Both | Ask the native detector for one symbol or all candidates it can decode. Native order is not a stable sorting key. |
| Mode `curved` | Standard | Use native curved-surface decoding. ArUco plus curved is rejected before native work. |
| Eps X / Y, Use alignment markers | Standard | Finder scan tolerances and alignment-marker refinement; unchanged for existing projects. |
| `qr_*` fields | ArUco | Seven finder-pattern/pyramid/timing controls. Rotation tolerance is in degrees between finder patterns, not a permitted overall label angle. |
| `aruco_*` fields | ArUco | All 35 writable native finder-detector fields, including threshold windows, contour/candidate filtering, canonical module decoding, corner refinement, ArUco3 and AprilTag extraction settings. |
| Minimum reads | Both | A successfully executed read below this count is an inspection failure, with reason `code_read_count_below_minimum`. |
| Maximum pixels | Both | Reject oversized input before QR grayscale conversion, selection and native decoding. Default 16,777,216; configurable up to 67,108,864. |

Corner refinement is a dropdown (`none`, `subpixel`, `contour`, `apriltag`). The
AprilTag option selects that native candidate extractor; its extra fields apply
only in that mode. ArUco3 must explicitly use `subpixel`, matching the refinement
the native implementation requires. ArUco3 canonical-size fields are ignored by
native ordinary detection. The inverted-marker flag applies to finder detection;
it does not guarantee that the full inverted QR payload can decode.

Threshold windows must be odd, ordered, and at least three pixels. Their range
and step may request at most 128 window sizes. Perimeter minima must not exceed
maxima. Fractions, angle conversions and integer limits are validated before
processing; the inspector puts rules in tooltips and identifies invalid fields.
Full names, defaults and ranges appear in the
[generated parameter reference](block-parameters.md#read-qr-codes-inspectread_qr).
Controls belonging to the other backend retain their saved values but do not
configure the selected native detector. Select the backend and mode deliberately
when comparing results.

## Images, selection and results

The working input must be uint8 gray or supported RGB/BGR(A). Tagged color is
converted to grayscale without changing upstream pixels. Explicitly convert
other depths upstream; the reader does not silently narrow high-depth images.
Each invocation creates its own native detector. Independent reader instances
can run on parallel branches, and last-input previews have no acquisition or
model state to update.

Optional mask and region inputs intersect. Pixels outside the selection become
white in a private search image; the source grid is not cropped or translated.
Keep the complete symbol and its quiet zone inside the selection. An empty
selection skips native decoding and reports zero reads. Native discovery on a
partially masked symbol is not proof that its whole outline lies inside the ROI.

Each valid code keeps its full text, `QR_CODE` symbology, four native camera-space
corners and their mean center. Undecoded candidates are excluded from `codes`
and do not satisfy Minimum reads. Multi-code ordering can differ between the
backends, so match by text/geometry or use a sorting tool when order matters.
Start/finish, tickets, judgments, diagnostic exceptions and downstream gating use
the common block runtime. The reporting event identifies `qr_backend`.

Text, outline and two image-plane axes are drawable result objects; they also
appear on the overlay and matching filmstrip images. These axes describe code
orientation in pixels, not a calibrated 3D robot pose. Disable **Draw overlay**
and optional result graphics to avoid their drawing overhead.

`rectified` emits only when exactly one symbol decodes and native decoding
provides one nonempty binary module image. Zero/multiple reads emit no single
rectified image. Module coordinates get a separate spatial root while retaining
the acquisition ticket. Camera-space `codes` and `overlay` retain the input grid.
The module image is not a camera crop or a complete source-resolution perspective
warp. Exceptions produce no current outputs and are contained by the executor;
a later valid invocation can recover. The pixel cap does not bound all native
allocations, candidate count or execution time, nor runtime/editor image copies.

## Qualified comparisons

Six original MIT scenes use known payloads and module bytes, deterministic
clutter, a 27-degree rotation, two-pixel modules, three differently oriented
labels, a left-symbol selection and a no-code scene. Known rendering transforms
provide independent camera-corner references. On Python 3.14.3/OpenCV 5.0.0 contrib
5.0.0.93, decoded corners are within 2.5 pixels of those references and every
emitted single rectified patch exactly matches the original module bytes.

Both backends read the ordinary/rotated/multiple/selected cases. ArUco reads the
small label while standard reports zero and fails its one-read criterion on this
build. The blank comparison explicitly allows zero reads. This is a recorded
fixture difference, not a general ranking of backend robustness. Printed labels,
production optics/lighting and quality grading require separate qualification.
For multipart payloads, use the separately qualified
[structured-append reader](qr-structured-append.md) and its explicit integrity limits.

Native references: [QR/ArUco detector](https://docs.opencv.org/5.0/main_modules/classcv_1_1QRCodeDetectorAruco.html),
[QR controls](https://docs.opencv.org/5.0/main_modules/structcv_1_1QRCodeDetectorAruco_1_1Params.html),
and the [pinned OpenCV 5.0 implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/src/qrcode.cpp).
