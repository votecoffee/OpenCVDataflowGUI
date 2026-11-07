# Matrix structure and regularization

Open [matrix_structure.json](matrix_structure.json) with **File → Open…**
and click **Run once**. The graph starts with the upper triangle of a
3 × 3 matrix and uses zeros as placeholders below the diagonal.
**Complete symmetric matrix** mirrors the upper triangle. **Create diagonal
matrix** builds a scaled identity, which **Multiply matrices** adds to the
symmetric matrix as a 0.5 diagonal regularizer.

**Measure matrix diagonal** reports `[4.5, 5.5, 6.5]` and a trace of
16.5. **Invert matrix** uses Cholesky to invert the regularized matrix.

The tools are under **Image processing → Statistics**. Matrices use nested
lists of finite numbers on data ports. A diagonal can be an editable or
wired vector. Its length must match the smaller matrix dimension. Symmetric
completion accepts a square matrix and copies the selected triangle while
preserving the diagonal.
[OpenCV array operations](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run "examples/opencv tools/matrix_structure/matrix_structure.json"
```

[All examples](../../README.md)
