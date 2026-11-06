# Barcode reading

The standard toolbox includes OpenCV QR and linear barcode readers. Install
the optional backend for Data Matrix, Code 128, PDF417 and more symbologies.

```powershell
python -m pip install -e ".[barcodes]"
```

Open the [Data Matrix example](../examples/inspection/datamatrix_reading/datamatrix_reading.json), or run it
from the repository root with:

```powershell
opencv-dataflow-run examples/inspection/datamatrix_reading/datamatrix_reading.json
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

Enable **Report decode errors** to collect invalid detections separately.
They do not satisfy the minimum read count or enter the valid codes list.
The reader decodes and reports backend errors. Barcode quality grading and
validation under production lighting and optics remain separate work.
See the [ZXing-C++ Python wrapper](https://github.com/zxing-cpp/zxing-cpp/tree/master/wrappers/python)
and [OpenCV barcode tutorial](https://docs.opencv.org/5.0/tutorials/objdetect/barcode_detect_and_decode/barcode_detect_and_decode.html).
