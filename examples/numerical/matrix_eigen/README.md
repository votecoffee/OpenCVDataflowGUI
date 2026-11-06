# Eigendecomposition

Open [matrix_eigen.json](matrix_eigen.json) with **File → Open…** and click
**Run once**. The upper branch builds a two-channel Gram matrix from four
measurements and finds eigenvalues `[16, 4]`. **Multiply matrices** projects
the observation `[2, 2]` onto the eigenvector axes. Its magnitude is about
`2.8284` on the dominant axis and zero on the second axis.
The lower branch shows a nonsymmetric matrix with eigenvalues `[0.8, 0.5]`
and a spectral radius of `0.8`.

Find **Eigendecomposition** under **Image processing → Statistics**. Enter
or wire a real square matrix of order 1 to 128. **Auto** uses the symmetric
solver when opposite entries match exactly. **General real** uses the
nonsymmetric solver and rejects spectra with complex eigenvalues.
Eigenvalues are in descending algebraic order. The corresponding eigenvectors
are rows with unit length. Repeated eigenvalues can have several valid
choices of eigenvectors. [OpenCV eigenpair reference](https://docs.opencv.org/5.0/main_modules/core_array.html#eigen)

**Basis rank** counts independent eigenvector directions at a relative
tolerance of `1e-10`. A value below the matrix order indicates the vectors
do not form a numerically independent basis. **Maximum relative residual**
compares `‖Av − λv‖₂` with `‖A‖F + |λ|` for each unit eigenvector. If a
residual exceeds the limit, the block fails and emits the residual
diagnostics. Set the limit to zero to disable that judgment.

```powershell
opencv-dataflow-run examples/numerical/matrix_eigen/matrix_eigen.json
```

[All examples](../../README.md)
