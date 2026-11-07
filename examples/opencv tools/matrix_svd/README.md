# Singular value decomposition

Open [matrix_svd.json](matrix_svd.json) with **File → Open…** and click
**Run once**. The first block decomposes a four-row measurement matrix.
The singular values are `[5, 2, 0]` and the numerical rank is 2. Keeping
one term retains about 86.21% of the squared Frobenius norm. The relative
Frobenius error estimated from the discarded spectrum is about 37.14%.
**Multiply matrices** applies the approximation to `[[2], [1], [3]]`
and returns approximately `[[10], [0], [0], [0]]`.

Find **Singular value decomposition** under **Image processing → Statistics**.
Set **Matrix** or wire a numeric table. For an m × n table, Thin returns an
m × r left basis and an r × n transposed right basis, where r is the smaller
dimension. Full returns m × m and n × n bases. Values only emits the
spectrum, numerical rank, and energy and error metrics.
Inputs are limited to two million values, and the smaller dimension is
limited to 128. Full bases are limited to two million values combined.

**Components** controls the reconstruction and metrics while leaving the
factor outputs intact. Zero keeps all terms. **Relative rank tolerance**
sets the cutoff as a fraction of the largest singular value. Zero selects
the automatic float64 cutoff. Singular vector signs and bases in repeated
singular subspaces may vary. The spectrum preview is available when result
graphics are enabled. [OpenCV SVD reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1SVD.html)

```powershell
opencv-dataflow-run "examples/opencv tools/matrix_svd/matrix_svd.json"
```

[All examples](../../README.md)
