# Checkerboard capture quality

Open [chessboard_quality.json](chessboard_quality.json) with **File → Open…**
and click **Run once**. The graph compares a sharp checkerboard against a
blurred copy using the same detected corners. Compare **Max width** in the
two check blocks. The blurred branch uses an 8 px limit to display both
measurements on a passing run. Set its **Max transition pixels** to 3 to
observe rejection and the withheld image output.

Find **Check chessboard quality** under **Calibration → Grid and coordinates**.
Set the number of inner corners. Leave **Corners** disconnected to detect
them with OpenCV's SB detector. A wired list must use the same captured
image's coordinate frame and SB corner order. **Orientation** chooses one
or both checkerboard edge directions. **Rise distance** of 0.8 measures the
10% to 90% intensity transition. [OpenCV checkerboard sharpness reference](https://docs.opencv.org/5.0/main_modules/objdetect_qrcode.html#estimatechessboardsharpness)

The report gives the mean transition width, black level, white level, and
sampled edges by direction. Brightness and contrast use 8-bit grayscale
units. Acceptance applies the limits to each requested direction's
averages. The overlay colors individual edges by the same limits.
Only accepted images are forwarded, with their original encoding and ticket.

The bundled synthetic fixture uses 32-pixel squares at grayscale levels 24
and 232, a light margin, and a 0.6-pixel Gaussian blur.

```powershell
opencv-dataflow-run "examples/custom tools/chessboard_quality/chessboard_quality.json"
```

[All examples](../../README.md)
