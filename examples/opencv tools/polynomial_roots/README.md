# Solve polynomial roots

Open [polynomial_roots.json](polynomial_roots.json) with **File → Open…**
and click **Run once**. The first block fits a quadratic to offsets measured
at positions 0, 1, and 2. It wires the fitted coefficient column
`[1, -4, 3]` into **Solve polynomial roots**. The **Analytic real** branch
finds the two positions where the offset is zero, approximately 1 and 3.
The quartic branch solves `x⁴ − 1 = 0` and shows all four roots as
`[real, imaginary]` pairs.

Find the block under **Image processing → Statistics**. Enter real
coefficients as a vector, row, or column. **Highest first** starts with the
coefficient of the highest power, and **Constant first** starts with the
constant term. A wired vector takes precedence. **All roots** works through
degree 32 and uses **Max iterations** to bound the search. **Analytic real**
returns distinct real roots for polynomials of degree three or lower.
For the iterative method, **Imaginary tolerance** controls which candidates
are also listed in **Real roots**.

The relative residual divides the magnitude of the polynomial at the
candidate by the sum of the magnitudes of its individual terms. A residual
above **Residual limit** marks the result as failed and keeps the candidates
available for inspection.
[OpenCV solveCubic](https://docs.opencv.org/5.0/main_modules/core_array.html#solvecubic)
and [OpenCV solvePoly](https://docs.opencv.org/5.0/main_modules/core_array.html#solvepoly)

```powershell
opencv-dataflow-run "examples/opencv tools/polynomial_roots/polynomial_roots.json"
```

[All examples](../../README.md)
