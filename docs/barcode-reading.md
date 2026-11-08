# Barcode reading

The standard toolbox includes OpenCV QR and linear barcode readers. Install
the optional backend for Data Matrix, Code 128, PDF417 and more symbologies.

```powershell
python -m pip install -e ".[barcodes]"
```

Open the [Data Matrix example](../examples/applications/datamatrix_reading/datamatrix_reading.json), or run it
from the repository root with:

```powershell
opencv-dataflow-run examples/applications/datamatrix_reading/datamatrix_reading.json
```

The example loads a generated Data Matrix, reads its lot code and checks for
the exact value `LOT-0042`. For your own graph, add **Inspection > Code
reading > Read barcodes (ZXing)** and choose the expected format or a format
group. The block defaults to Data Matrix. Optional masks and regions
restrict the search while retaining coordinates in the original image.
Keep the symbol and its quiet zone inside the selection.

Valid reads report text, symbology, the AIM identifier reported by the
backend, raw payload bytes in hexadecimal, content type, orientation and
corners. The default HRI text mode formats GS1 human-readable text. Use the
raw bytes when the application must preserve payload encoding. **Verify
decoded text** accepts the `codes` output.

OpenCV QR, OpenCV linear, and ZXing readers can draw the decoded text, symbol
outline, and two pose axes centered on each valid code. The axes follow the
detected symbol edges. The **Draw overlay** setting controls the reader's
image output. The same objects are available as **Result graphics** and appear
on matching filmstrip images when result graphics are enabled. Display labels
may shorten long payloads or escape non-ASCII characters. The `codes` output
keeps the complete decoded value.

**Read QR codes** also emits **Rectified** when exactly one symbol decodes
and OpenCV returns its binary module image. This small image has its own grid
and keeps the camera ticket. Use **Overlay** or **Codes** for camera-space
geometry. With zero or multiple decoded symbols, Rectified emits no image.

Enable **Report decode errors** to collect invalid detections separately.
They do not satisfy the minimum read count or enter the valid codes list.
The reader decodes and reports backend errors. Barcode quality grading and
validation under production lighting and optics remain separate work.
See the [ZXing-C++ Python wrapper](https://github.com/zxing-cpp/zxing-cpp/tree/master/wrappers/python)
and [OpenCV barcode tutorial](https://docs.opencv.org/5.0/tutorials/objdetect/barcode_detect_and_decode/barcode_detect_and_decode.html).
