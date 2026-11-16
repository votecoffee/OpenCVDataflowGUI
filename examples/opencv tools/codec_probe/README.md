# Image file and codec preflight

Open a project with **File → Open** and use **Run once**. Select a probe to
inspect **Info**, reader/writer availability, page count and **OK**. No plugins,
downloads, source acquisition tickets or destination-image writes are required.

| Project | Expected result |
| --- | --- |
| [File inventory](file_inventory.json) | One Unicode-named PNG, three mixed-depth TIFF pages and three APNG frames. |
| [Encoder support](encoder_support.json) | PNG, TIFF and BMP registry checks; no file is written to test them. |
| [Page policy](page_limit.json) | Deliberate preflight FAIL: the three-page TIFF is below the configured minimum of four. The actual count remains available. |
| [Invalid header](invalid_header.json) | Deliberate preflight FAIL: the PNG signature is recognized, but native header inventory returns zero. |
| [Unavailable encoder](unsupported_encoder.json) | Deliberate preflight FAIL for the invented `.notacodec` extension. |

Failures retain structured reason codes/messages; they are not processing
exceptions. Passing a signature/count check does not validate compressed pixels.
Registered encoders do not guarantee that a particular image type or codec option
can be written. Unrequested/unperformed optional observations have no output event.

Original synthetic assets/projects: **OpenCVDataflowGUI contributors, MIT**.
No customer or third-party images are included. Every canvas contains attribution
and the expected results. [truth.json](truth.json) records counts and asset hashes.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/codec_probe/build_example.py"
.venv/Scripts/python.exe -m opencv_dataflow.assistant_cli run "examples/opencv tools/codec_probe/file_inventory.json" --summary
```

See [the snapshot and capability contract](../../../docs/codec-preflight.md).
