# Completed laser-profile review

Load [plugin.py](plugin.py) explicitly in OpenCVDataflowGUI, open a project below,
run in **Debug**, then use **Review completed laser profile…** in the laser block's
inspector. The bundled dev3 wheel is checksum-pinned and loaded without installing
or updating the host environment. Restart before switching an already loaded version.

The independent Results popup synchronizes the source overlay, trend, sample table
and calibrated native cloud. Shift-click source or click trend, select a row or use
the source-bin control. Wheel zoom/drag pan, Focus/Overview and availability filters
change the view only. The native calibrated cloud supports fly-around, sample
selection and separately labeled derived ray-segment/plane/axis layers. Pixel-only,
missing and rejected bins cannot acquire metric points through review.

| Project | Preserved outcome |
| --- | --- |
| [Calibrated step](../../inspection/calibrated_step.json) | Generated known-geometry control: 480/480 available, independent nominal 50-mm step. Not a physical gauge qualification. |
| [Unavailable bins](../../inspection/calibrated_unavailable.json) | Fail: 390/480; missing, competing and input-ceiling samples remain excluded. |
| [Entirely missing](../../inspection/calibrated_missing.json) | All-zero input control: 0/480 and no stale cloud. Original good calibration retained. |
| [Geometry rejected](../../inspection/geometry_rejected.json) | Deliberately incorrect plane puts intersections behind the camera: accepted pixel peaks, no accepted metric samples/cloud. This is a labeled negative calibration control, not a new calibration estimate. |
| [Graphics off](../../inspection/graphics_off.json) | Same numeric points/report; emitted overlay/chart absent and cloud colors omitted. Review-only images use frozen decisions. |
| [Real WLI 1](../../inspection/real_wli1.json) | Fail: 72/120; pixel-only, input-ceiling rejection retained. |
| [Real WLI 2](../../inspection/real_wli2.json) | Fail: 42/120; missing/ambiguous data retained. |
| [Real WLI 3](../../inspection/real_wli3.json) | Fail: 2/120; clipping and poor reference agreement retained. |
| [Real WLI 4](../../inspection/real_wli4.json) | Fail: 7/120; clipping and poor reference agreement retained. |

All nine use the same **90% availability** gate. The existing models, learned stripe
settings and ROI remain unchanged except the explicitly bad-plane control; graphics-off
changes display generation only. The selected real JPEGs are unchanged. No thresholds
were tuned to improve test images. Public industrial-software centerline references
are included in [sources/centerline](sources/centerline), separately from training.
They do not establish metric accuracy, stripe identity or completeness.

Six [frozen exports](snapshots) retain the exact executed `.slp`, native source and
consumed images, profile/stripe NPY arrays, optional cloud arrays/PLYs, full JSON and
readable source/overlay/trend/assessment PNGs. NPY preserves native precision and NaNs;
PLY and PNG are host/display representations. Review does not extract stripes,
undistort pixels, intersect rays, recalibrate, reload a recipe or change teaching.
Unstored ray/reprojection diagnostics remain unavailable. A new publication marks an
open review stale; capture the latest completion explicitly. Training Apply/Cancel
and unapplied drafts stay independent.

See [attribution and scope](SOURCES.md), [workflow](COMPLETED_REVIEW.md),
[verification](verification.json), [wheel proof](wheel-verification.json) and
[file pins](integrity.json). Models and inputs are shared by content hash across
projects. Real WLI captures have no supplied camera/laser calibration and remain
pixel-only. Availability passes and imported calibration are not metrology acceptance.

Replay through the host's existing Python environment, without installing anything:

```powershell
python examples.py verify . C:\path\to\new-review-check --native
```

This checks file pins, numeric/image/cloud parity, native Results controls, six
frozen exports and Save As/reopen with byte-identical recipe archives.
