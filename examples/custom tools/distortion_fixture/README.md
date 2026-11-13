# Controlled image distortion fixtures

Open `project.json` and **Run once**. Four branches distort an original 16-bit
RGB part using shear, perspective, lens-like coefficients and seeded bounds.
They reuse the generated maps through ordinary **Remap image coordinates**
blocks and compare with independently calculated references. All four checks
must report **maximum absolute error 0**. Serial and parallel execution repeat
the same results. No camera, trained matcher or download is needed.

**Generate image distortion fixture** is under **Custom > Image processing >
Test fixtures**. Inspect **Mapping** for effective coefficients, normalization,
source/output frames and map precision. **Point pairs** reports output pixel
positions and their analytic input donors; **Map X/Y** are float32 planes of
those donors on the output grid. These are output-to-input maps, suitable for
OpenCV remapping. They are not an estimated or automatically inverted transform.
Green numbered markers identify output sampling probes.

The new image has its own coordinate frame. The source pyramid/frame context
is retained inside the mapping record; an old scale transform cannot represent
this nonlinear warp. Typed maps preserve the same output frame through the
floating and packed remappers and reject swapped/mixed/cropped grids or a
different source frame. Packing quantizes donors to OpenCV's fixed-point grid;
its pixel values can differ from floating-map sampling. Keep the typed packets
intact when reusing maps; plain image files do not preserve this provenance.

**Valid mask** is conservative source support. Optional masks/regions filter
that support while the entire image is still warped. Nearest sampling requires
in-bounds donor centers; linear/cubic/Lanczos4 additionally reserve source halos
of 1/2/4 pixels. Reflected or replicated pixels never count as original support.
All image channels, including alpha, are resampled independently. This is not
alpha-aware compositing. Float64 cubic remap is rejected on the supported build
because native boundary sampling truncates fractions; use linear or float32.

Fixed mode uses signed coefficients. Uniform-bounds mode draws each coefficient
in the fixed documented order from `[-bound,+bound]`, using nonnegative bounds
and a private PCG64 generator. A wired seed overrides the parameter; the same
seed/configuration repeats without advancing global RNG state. Unsafe horizons,
nonpositive sampled Jacobians and excessive donor ranges fail explicitly.
Positive sampled Jacobians do not establish global invertibility.

Images, independent matrix/polynomial references, generator and graph are
original content under the repository's MIT license. The golden images use
OpenCV resampling after independently calculating their donor maps.
`assets/truth.json` records continuous donor probes and coefficients.
Regenerate with `python "examples/custom tools/distortion_fixture/generate.py"`.
See the [distortion model documentation](../../../docs/distortion-fixtures.md)
and [OpenCV geometric transforms](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html).
