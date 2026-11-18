# Native ALIKED and LightGlue ONNX controls

Open a project below and press **Run once**. No downloads or optional Python ML
packages are needed for these original arithmetic controls. They exercise the
native APIs and result contracts; the tiny models contain no learned weights.

| Project | Expected result |
| --- | --- |
| `translation.json` | Seven score-qualified inliers; measured X translation 2.007843 pixels. |
| `score_rejection.json` | One retained match, seven rejected candidates; failed fit and no pose. |
| `selected.json` | Source ROI keeps six feature centers; six matches/inliers. |
| `empty.json` | Zero selected features; normal failed match with native inference skipped. |
| `high_depth.json` | Same registration from tagged RGB uint16; output pixels/depth unchanged. |
| `pair_budget.json` | Explicit attention budget error; no partial current output. |

Select **Report** for model hashes, candidates, rejected scores and cache reuse.
Select **Matches** for inlier flags; use **Overlay** or the filmstrip for drawable
result markers. The reference source is independent of the current query ticket.

The source red-channel average moves eight generated observations. Descriptor
agreement and normalized point displacement determine confidence. Thus the models
respond to inputs without pretending to be trained ALIKED or LightGlue networks.
The fixed candidate head can only represent these eight paired control rows.
`build_example.py` regenerates all assets/projects in a new directory and refuses
to overwrite an existing asset folder. [SOURCES.md](SOURCES.md) records attribution.

For the separate pretrained reference, explicit local-only preparation is:

```powershell
.venv\Scripts\python.exe "examples/opencv tools/native_lightglue/prepare_reference.py" --download
```

This verifies exact hashes, preserves differing existing files, and creates
`.local/native_lightglue_reference/reference.json` plus a numerical qualification
receipt. Open that local project to inspect **211 matches / 207 inliers** on the
tested CPU build. Rerun without `--download` to verify already prepared assets.
External assets stay ignored: the export repository has no explicit license file,
so these bytes are not included in Git. Review their applicable upstream rights
before any redistribution. No network access occurs while a graph executes.

See [block contracts and limits](../../../docs/native-lightglue.md). The raw
controlled feature output can be reused for downstream inspection, while actual
model quality and application accuracy need separate representative validation.
