# Optical-flow file round trips

Open a project using **File → Open**, then **Run once**. No plugin or download is
needed. The writer creates fresh UUID-named `.flo`/`.flo.json` pairs under the
ignored `output` folder, so repeated runs preserve earlier files. **Save As** can
place new output files beside your own saved copy; original input assets remain
referenced.

| Project | Expected behavior |
| --- | --- |
| [Round trip](roundtrip.json) | Load analytic dx=3, dy=-2 → save native field plus image-pair manifest → reload/verify both images → warp, with exactly 11,750 covered pixels matching the target. |
| [Computed motion](computed_motion.json) | Same transport for a Farneback field measured against independent analytic motion; field samples survive exactly. |
| [Bare interoperability](bare_file.json) | Load ordinary `.flo` without a companion; direction/pair remain caller declarations. |
| [Wrong direction](direction_mismatch.json) | Deliberately invalid Warp input: configured direction contradicts the file manifest; no image emission. |
| [Wrong image](wrong_image.json) | Deliberately invalid Warp input: Target is substituted for Source; no image emission. |

Select a loader/writer to inspect samples, hash, invalid counts, association and
the direction/magnitude diagnostic image. Select Warp for the result image,
coverage and the list of checked image identities. Lean skips optional flow
visualization. [truth.json](truth.json) records all asset hashes and measured
interior errors. On this fixed original fixture the Farneback median endpoint
error is about **0.00018 pixels**, and the 95th percentile about **0.00243 pixels**.
These measurements describe this synthetic translation only.

All images, fields and projects are original **OpenCVDataflowGUI contributors,
MIT**. No customer or third-party images are included. Every canvas includes an
attribution/scope comment. Fingerprints identify image contents/grids; they do not
prove capture identity, camera timing, physical calibration or motion accuracy.

```powershell
.venv/Scripts/python.exe "examples/opencv tools/flow_file_roundtrip/build_example.py"
.venv/Scripts/opencv-dataflow-api.exe run "examples/opencv tools/flow_file_roundtrip/roundtrip.json" --summary
```

See [file/manifest contracts](../../../docs/flow-files.md) for precision controls,
invalid-vector policies, ticket behavior and publication/rollback limits.
