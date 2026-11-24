# Tesseract OCR on printed labels

Open [tesseract_ocr.json](tesseract_ocr.json) and choose **Run Once**. The first
block finds three horizontal text regions. Tesseract reads **PUMP 7 VALVE 24 LOT
305**, emits word confidences and source-image boxes, and draws the words on
the filmstrip result.

The six registered boxes overlap the [reference glyphs](truth.json) by
96.9% to 100%. Raw word confidences range from 95.37% to 96.65% with Tesseract
5.5.1. The example uses LSTM recognition and the single-line layout.

## Tesseract runtime

Install Tesseract and put its executable on the system PATH, or set
`OPENCV_DATAFLOW_TESSERACT` to its absolute path before launching the app.
On Windows, the repository's `.venv` also discovers a Pixi environment
installed under `.local/tesseract`. The following commands create that
environment from the repository root.

```powershell
pixi init .local/tesseract --channel conda-forge --platform win-64
pixi add --manifest-path .local/tesseract/pixi.toml "tesseract==5.5.1"
```

The app uses Tesseract's command-line TSV output to retrieve per-word
confidence and geometry on the stock OpenCV wheel. The Python OpenCV
`OCRTesseract.run` binding exposes text alone.
[Tesseract command-line reference](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)

The embedded [English data](assets/tessdata/eng.traineddata) comes unchanged
from [tessdata_fast 4.1.0](https://github.com/tesseract-ocr/tessdata_fast/tree/65727574dfcd264acbb0c3e07860e4e9e9b22185)
and is distributed under the [Apache 2.0 license](assets/tessdata/LICENSE-APACHE-2.0.txt).
The example pins SHA-256
`7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`.
The chart is original and MIT-licensed.

## Native adapter qualification

[qualify_native.cpp](qualify_native.cpp) exercises the C++ OpenCV 5.0.0
Tesseract adapter, including word-level confidences and boxes. The test
compiled the upstream adapter against Tesseract 5.5.1 and ran it on the same
padded crops and pinned English data. Its six boxes and recognized strings
matched the app, with normalized confidences agreeing within `1e-7`.
The harness accepts a tessdata directory and one or more binary 8-bit PGM crops.
[OpenCV OCRTesseract reference](https://docs.opencv.org/5.0.0-alpha/d7/ddc/classcv_1_1text_1_1OCRTesseract.html)

`python build_example.py --output <directory>` recreates the chart, truth
and saved graph. Copy the `assets/tessdata` folder with the new project.
