# Hit-or-miss binary patterns

Open a project with **File → Open** and press **Run once**. No plugins or downloads
are required. Select a pattern block and click **Edit kernel…** to edit foreground
(`1`), background (`-1`) and ignored (`0`) cells. **Run block** previews the draft
using the last input; **Apply configuration** saves the edits.

| Project | Expected result |
| --- | --- |
| [Binary patterns](binary_patterns.json) | Three isolated pixels, two cross junctions, two right endpoints and one north/east corner in parallel branches. |
| [Dark patterns](dark_patterns.json) | Inverted pixels and Dark polarity find the same three isolated pixels and two junctions. |
| [Selected centers](selected_centers.json) | A rectangular mask and circular region intersect, retaining only `(36,12)`. Both use the source image's frame. |
| [Border support](border_support.json) | Default full support finds three isolated pixels; native neutral padding adds the pixel at `(0,0)`. |
| [Count failure](count_failure.json) | Deliberate inspection FAIL: requires four isolated pixels, finds three, emits the full mask/coordinates and a fail reason. |
| [Invalid binary](invalid_binary.json) | Deliberate validation ERROR: a gray value of 127 in strict binary mode. No processing outputs. |

Results include the mask, total count, coordinates, measurements and optional
drawable markers. Coordinates identify kernel anchors in the input image.
Counts are matching pixels, not connected parts. Change kernels or limits to
explore the behavior. These exact local patterns do not handle arbitrary part
rotation, scale or grayscale texture.

Original synthetic assets/projects: **OpenCVDataflowGUI contributors, MIT**.
No customer or third-party images are used. Every canvas includes attribution
and expected coordinates. [truth.json](truth.json) records exact kernels,
coordinate truth and asset hashes. These one-pixel fixtures demonstrate the
tool; they are not physical inspection data.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/hit_or_miss/build_example.py"
.venv/Scripts/python.exe -m opencv_dataflow.assistant_cli run "examples/opencv tools/hit_or_miss/binary_patterns.json" --summary
```

[Detailed contract](../../../docs/hit-or-miss.md) · [All OpenCV examples](../README.md)
