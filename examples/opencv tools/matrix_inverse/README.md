# Invert matrix

Open [matrix_inverse.json](matrix_inverse.json) with **File → Open…** and
click **Run once**. The upper branch uses LU to invert a square calibration
matrix with determinant 5. **Multiply matrices** applies that inverse
to two mixed observations and recovers `[[2, 4], [3, 6]]`.

The lower branch uses SVD on a rectangular matrix whose two columns
describe the same signal. The block reports rank 1 and a reciprocal
condition of 0. Applying the pseudoinverse to `[[4], [8], [12]]` returns
the minimum-norm signal pair `[[2], [2]]`.

Find **Invert matrix** under **Image processing → Statistics**. Enter or
wire a rectangular table of real numbers. LU requires a full-rank square
matrix. Cholesky also requires symmetry and positive definiteness. SVD
keeps singular values above the largest singular value times float64
machine epsilon times the larger matrix dimension.

**Reciprocal condition** is the smallest singular value divided by the
largest for full-rank inputs, and zero for numerical rank deficiency.
The determinant port emits a value for square inputs when the determinant
fits in float64. The inverse is available independently.
[OpenCV invert reference](https://docs.opencv.org/5.0/main_modules/core_array.html#invert)
and [OpenCV determinant reference](https://docs.opencv.org/5.0/main_modules/core_array.html#determinant)

```powershell
opencv-dataflow-run "examples/opencv tools/matrix_inverse/matrix_inverse.json"
```

[All examples](../../README.md)
