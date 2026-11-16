# QR backend comparisons

Open a project below with **File → Open**, then **Run**. The standard and ArUco
branches read the same source image. Inspect Codes/Count and compare their result
images; double-click a result image to enlarge it. The source uses its filename
as the ticket. No camera, plugin, model download or optional dependency is needed.

| Project | Expected result on OpenCV 5.0.0 contrib 5.0.0.93 |
| --- | --- |
| [Clutter](cluttered.json) | Both read `CLUTTER-42` among surrounding shapes and background texture. |
| [Rotation](rotated.json) | Both read `ROTATED-42`, placed at 27 degrees. |
| [Small label](small.json) | ArUco reads `SMALL-07` at two pixels/module; standard returns zero and deliberately fails its one-read criterion. The graph therefore has a failed branch. |
| [Multiple labels](multiple.json) | Both read `LEFT-42`, `CENTER-07`, `RIGHT-18`; native ordering can differ. |
| [Selected label](selected.json) | A rectangle keeps only `LEFT-42`; reported corners stay on the original image. |
| [No label](blank.json) | Both return zero codes; Minimum reads is explicitly zero so this comparison passes. |

Read QR codes stays in **OpenCV → Object Detection (objdetect) → Code reading**.
Its Backend dropdown preserves the standard default for existing projects.
The same reader exposes all seven native QR/ArUco controls and all 35 native
finder-detector fields, including a corner-refinement dropdown. Settings for the
other backend retain their values but are inactive. The small example sets the
ArUco minimum module size to two. See [control and result contracts](../../../docs/qr-backends.md).

Overlay/result graphics contain the payload text, outline and two image-plane
axes. Rectified is a binary module image with its own grid, emitted only for
one valid read. It is intentionally absent for the three-code and blank cases.
Disable Draw overlay plus optional result graphics for less drawing work.

Run a project without the editor from the repository root:

```powershell
opencv-dataflow-api run "examples/opencv tools/qr_backends/rotated.json" --summary
```

All payloads, images, placement references and projects are original work by
OpenCVDataflowGUI contributors under the repository's MIT license. Native
OpenCV QRCodeEncoder generates the symbols; no third-party/customer media is
included. Every project contains an attribution/comment block. `truth.json`
records complete payloads, module/pixel hashes, known rendering-transform
corners and observed reads by backend. `build_example.py` regenerates them:

```powershell
python "examples/opencv tools/qr_backends/build_example.py"
```

These focused synthetic fixtures demonstrate tool behavior. They do not rank
the backends across arbitrary images or qualify print quality, production
lighting/optics, physical pose, or ordered structured-append recovery.
