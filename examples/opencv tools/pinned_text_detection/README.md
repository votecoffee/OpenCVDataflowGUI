# Pinned PP-OCRv3 text detection

Prepare the model once from the repository root using the app's environment:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/pinned_text_detection/prepare_model.py"
```

This explicitly downloads 2.4 MB of Apache-2.0 weights and verifies their exact
size/hash. It seals preprocessing and CPU execution settings in the ignored
`.local/ppocr_en` folder. Opening or running a project never downloads a model.
`--model <path>` can reuse the identical already-downloaded artifact.

Open `labels.json` and run once. Select **Text regions from probability map** to
inspect its source-aligned regions and report. The inference block retains the
736x736 named probability map and an execution receipt. Region outlines appear
on the source image and in the filmstrip. This step detects regions; it does not
read the printed text. Connect the regions to a separately configured CTC
recognition block for OCR.

| Project | Expected result |
| --- | --- |
| `labels.json` | Pass; three industrial-style text labels |
| `rotated.json` | Pass; three labels rotated by 12 degrees |
| `selected_roi.json` | Pass; only the first label's center is in the ROI |
| `joined_words.json` | Pass; one contiguous text token |
| `split_words.json` | Pass; two spatially separated tokens |
| `empty.json` | Fail; a blank image has no regions |
| `count_failure.json` | Fail; three regions when at least four are required |
| `truncated.json` | Fail; three accepted regions, only the best one retained |

These original MIT synthetic label fixtures have independent source glyph bounds
and transformed polygons in `truth.json`. They contain no customer images.
The pinned PaddlePaddle/OpenCV Zoo model is unchanged; attribution is also
embedded in every project. See [SOURCES.md](SOURCES.md) and
[MODEL-LICENSE.txt](MODEL-LICENSE.txt).

Debug runs include optional raster previews and drawable outlines. Lean skips
those graphics while retaining numerical regions and judgments. The original
image's format, depth, ticket and frame are preserved by the decoder. Its ROI
filters use region centers; they do not clip contours or constrain inference.

`build_example.py` regenerates graphs, images and reference annotations in a
fresh directory using the normal app dependencies. It preserves different
existing assets. Save As copies the sealed model contract/weights and notices;
input images retain correctly rebased references. Copy this whole folder for
a self-contained transfer.

The controlled examples validate coordinate mapping, orientation, grouping and
failure behavior. Their accuracy is not an industrial OCR benchmark. Lighting,
font size, blur, natural backgrounds and field-specific word grouping need
separate image-based qualification. See [the full decoder contract](../../../docs/text-detection.md).
