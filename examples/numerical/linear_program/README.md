# Solve linear program

Open [linear_program.json](linear_program.json) with **File → Open…** and
click **Run once**. The two variables are continuous amounts assigned to
two activities.

- **Maximize value** allocates `[2, 2]` under a shared limit of 4 and
  individual limits of 2 and 3. The objective value is 10, and the
  constraint slacks are `[0, 0, 1]`. The solution is wired to
  **Multiply matrices**, which reports actual usage `[4, 2, 2]`.
- **Meet demand at minimum cost** uses the same individual limits and
  requires a total of at least 4. It allocates `[1, 3]` at a cost of 9.

Find **Solve linear program** under **Image processing → Statistics**.
Enter one objective coefficient per variable. Each constraint row ends
with an upper bound. For example, `[1, 1, 4]` represents x + y ≤ 4,
and `[-1, -1, -4]` represents x + y ≥ 4. The solver supports up to 64
nonnegative variables and 1,024 constraints. Wired inputs take precedence.

The **Slacks** output gives each bound minus actual usage. A status of
`unique` or `multiple` includes a solution. The block reports
`infeasible`, `unbounded`, or `numeric_failure` as a failed result
and emits the status alone. **Constraint tolerance** specifies the
allowed numerical violation in the constraints' original units.
[OpenCV solveLP reference](https://docs.opencv.org/5.0/main_modules/core_optim.html#solvelp)

```powershell
opencv-dataflow-run examples/numerical/linear_program/linear_program.json
```

[All examples](../../README.md)
