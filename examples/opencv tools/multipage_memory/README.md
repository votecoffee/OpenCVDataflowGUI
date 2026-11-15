# Multipage TIFF in memory

Open a project below with **File → Open**, then **Run all**. Three acquisitions
are batched into an ordered group before the in-memory encoder and decoder run.
No plugins, downloads or output files are required.

| Project | Expected result |
| --- | --- |
| [Typed round trip](typed_roundtrip.json) | Three mixed-size/depth pages, exact pixels, original tickets and coordinate frames. |
| [Raw byte round trip](raw_roundtrip.json) | Exact pixels using raw TIFF bytes; fresh page frames and a container ticket replace individual page context. |
| [Select strided pages](selected_pages.json) | Select original indexes 0 and 2; preview position 1 shows original page 2. |
| [Pixel budget guard](pixel_budget_guard.json) | Intentional decoder validation error before decoding pixels; no decoder output event. |
| [Predictor guard](predictor_guard.json) | Intentional encoder validation error: horizontal prediction cannot process the floating page. |

The passing projects require zero image-norm difference against original page 2
and unbatch all selected pages. Select **Decode multipage TIFF** to view its
preview and Info report. Change Preview page to inspect a different page; also
change the reference block's index if keeping the exact comparison connected.

The original MIT fixture contains 72×48 uint8 gray, 64×40 uint16 BGR and 56×32
float64 BGRA pages, including signed HDR samples and alpha. These synthetic
tool demonstrations are not physical acquisitions or ground-truth inspection
data. Float64 multichannel TIFF may print the native photometric/ExtraSamples
warning; sample bytes are checked exactly. See [the codec contract](../../../docs/multipage-memory.md).

Original images and projects: **OpenCVDataflowGUI contributors, MIT**. There are
no customer or third-party images. Every canvas includes this attribution and the
expected outcomes. [truth.json](truth.json) records dimensions and hashes.
[build_example.py](build_example.py) regenerates the asset and projects:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/multipage_memory/build_example.py"
.venv/Scripts/python.exe -m opencv_dataflow.assistant_cli run "examples/opencv tools/multipage_memory/typed_roundtrip.json" --all-images --summary
```
