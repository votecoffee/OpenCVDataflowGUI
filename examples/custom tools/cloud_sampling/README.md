# Point-cloud sampling

Open [the project](cloud_sampling.json) and run it once. Both branches
retain 20 of the 325 points in a colored PLY of a dense raised patch on a
more sparsely sampled plate. **Indices** lists the original row IDs in the
order they were selected. The sample preserves colors, normals, frame,
and length units.

With seed 71, the farthest-point branch covers the input with a maximum
nearest-sample distance of 160 mm. The random sample's independently
measured distance is 534.12 mm. This difference describes this fixture and
seed. In farthest-point mode, **Minimum spacing** can stop sampling when
the next candidate is closer than the configured distance in cloud units.
**Max distance evaluations** bounds the total search effort. Hitting this
limit reports a failure and retains the partial sample for inspection.

The installed OpenCV 5.0.0 wheel omits the native C++ sampling functions.
The block uses seeded NumPy row selection and a greedy farthest-point
search. [OpenCV sampling reference](https://docs.opencv.org/5.0/main_modules/_3d.html)

The synthetic fixture is MIT-licensed with the repository. Run
`build_example.py --output <empty-folder>` to regenerate the saved project and PLY.

```powershell
opencv-dataflow-run "examples/custom tools/cloud_sampling/cloud_sampling.json" --profile lean
```
