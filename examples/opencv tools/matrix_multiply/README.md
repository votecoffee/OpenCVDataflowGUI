# Multiply matrices

Open [matrix_multiply.json](matrix_multiply.json) with **File → Open…** and
click **Run once**. The first block fits two calibration curves. Its solution
matrix contains the intercepts `[2, 10]` and slopes `[3, -2]`. The second
block receives those coefficients through its **right** input and predicts
the two readings at positions 4, 5, and 6.

**Transpose left** turns the two configured rows into three rows of
`[1, position]`. **Transpose addend** and an addend weight of 0.5 add
`[1, -1]` to every prediction row. The output is approximately
`[[15, 1], [18, -1], [21, -3]]`.

Find **Multiply matrices** under **Image processing → Statistics**. It
computes a weighted product of two real numeric matrices and can add a
third weighted matrix of the same output shape. Enter matrices in the block
or wire them from upstream tools. Each matrix can be transposed independently.
The left column count must match the right row count after transposition.
The block uses float64 arithmetic and caps each matrix at two million values.
[OpenCV GEMM reference](https://docs.opencv.org/5.0/main_modules/core_array.html#gemm)

```powershell
opencv-dataflow-run "examples/opencv tools/matrix_multiply/matrix_multiply.json"
```

[All examples](../../README.md)
