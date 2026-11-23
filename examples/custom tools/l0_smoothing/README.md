# L0 image smoothing

Open [l0_smoothing.json](l0_smoothing.json) and choose **Run Once**. Compare the
Gaussian baseline with two L0 smoothing strengths on a textured 16-bit signal.
The known main step is 27,900 intensity levels, and the five-pixel stripe has
a contrast of 5,800 levels.

| Image | RMSE against the clean target | Main step | Stripe contrast |
| --- | ---: | ---: | ---: |
| Textured input | 1,269.8 | 27,935.3 | 5,745.6 |
| Gaussian baseline | 1,637.0 | 4,734.2 | 3,130.0 |
| L0, lambda 0.001 | 103.2 | 27,893.6 | 5,692.3 |
| L0, lambda 0.02 | 761.2 | 27,888.2 | 0.6 |

The main step averages the boundary difference between columns 82 and 83.
Stripe contrast compares the mean of columns 121–123 with columns 116–118 over
rows 26–60. The region branch uses the weaker smoothing strength inside the
rectangle and retains the original pixels outside it.

The block appears in **Custom → Image processing → Filters**. It uses SciPy
cosine transforms to solve the L0 gradient problem with zero outward
differences at the image borders. **Lambda weight** sets the cost of retaining
a gradient. **Kappa** controls how quickly the solver increases the gradient-fit
penalty. The block preserves sample depth and alpha. [Original L0 smoothing method](https://jiaya.me/file/archive/projects/L0smoothing/)

The native `ximgproc.l0Smooth` path for S4-40 remains open. The tested
OpenCV 5.0.0 wheel turned a simple two-level 8-bit step into an all-zero
image. The saved project uses the independently checked SciPy implementation.

The graph, builder and synthetic images are original MIT assets. Regenerate
them from the repository root with the example builder.

    python "examples/custom tools/l0_smoothing/build_example.py" --output "examples/custom tools/l0_smoothing"
