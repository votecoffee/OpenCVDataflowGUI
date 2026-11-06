# Solve linear system

Open [linear_system.json](linear_system.json) with **File → Open…** and click
**Run once**. The first block fits two calibration curves using the same four
sample positions. The second block re-solves the system from the fitted
observations received over a graph connection.

The coefficient matrix has one row per sample, with columns for the intercept
and sample position. The solution rows contain the fitted intercepts
`[2, 10]` and slopes `[3, -2]`. The first curve has residuals
`[1, -1, -1, 1]` and an RMSE of 1. The second curve and both verification
curves have an RMSE near zero. The thumbnail plots the residuals for the
first observation column.

Find **Solve linear system** under **Image processing → Statistics**.
Enter the coefficient table and a vector or table of observations, or wire
them from upstream blocks. Each observation column gives a separate solution.
QR fits systems with at least as many equations as unknowns and full column
rank. LU takes a full-rank square matrix. The eigen method requires a
symmetric, full-rank square matrix, and Cholesky also requires positive
definiteness. SVD returns a minimum-norm least-squares solution and supports
rank-deficient systems and systems with more unknowns than equations.
[OpenCV solve reference](https://docs.opencv.org/5.0/main_modules/core_array.html#solve)

```powershell
opencv-dataflow-run examples/numerical/linear_system/linear_system.json
```

[All examples](../../README.md)
