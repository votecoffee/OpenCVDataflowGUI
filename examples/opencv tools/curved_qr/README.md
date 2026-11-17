# Curved QR labels

Open a project and press **Run**. Select **Read QR codes** to inspect its
**Overlay**, **Rectified** and **Info** outputs. The filmstrip displays the text,
outline and axes. Double-click the result image to enlarge it. No camera,
plugin, optional barcode decoder or download is required.

| Project | Expected curved read | Flat-mode comparison |
| --- | --- | --- |
| [mild.json](mild.json) | Pass, exact original modules | Reads the same text. |
| [wrapped.json](wrapped.json) | Pass, tilted cylindrical label; exact modules | Misses the label. |
| [vertical.json](vertical.json) | Pass, quarter-turn wrap; up to one sampled module error | Misses the label. |
| [larger_symbol.json](larger_symbol.json) | Pass, version-3 symbol; exact modules | Misses the label. |
| [selected.json](selected.json) | Pass, whole-label region on the original source grid | Misses the label. |
| [severe.json](severe.json) | Deliberate failure: stronger curvature | Also misses. |
| [rotated_failure.json](rotated_failure.json) | Deliberate failure: 27-degree view | Also misses. |

Change **Mode** from `curved` to `single` to try the flat comparison, then change
it back. Minimum reads remains one, so a missed label is an explicit failed
judgment. The two negative projects intentionally return a nonzero CLI exit.

```powershell
python -m opencv_dataflow.assistant_cli run "examples/opencv tools/curved_qr/wrapped.json" --summary
```

The outline is OpenCV's approximate detection quadrilateral. It does not trace
the curved boundary; its fourth corner can differ from the actual surface
corner by more than 40 pixels. Axes show image-plane orientation, not a physical
pose. Rectified modules have a separate observation grid with no native inverse
map to the source. Correct text does not guarantee an error-free sampled module
image. See [geometry contracts](../../../docs/curved-qr.md).

Original MIT images, renderer, analytic references and projects by
OpenCVDataflowGUI contributors. No customer, upstream or third-party image
assets. Each canvas carries attribution. `truth.json` records pinhole/cylinder
parameters, analytic corners, expected text, original module matrices/hashes,
image hashes and deliberate misses. `build_example.py` regenerates the collection
in a fresh folder and refuses to replace different existing image bytes.
