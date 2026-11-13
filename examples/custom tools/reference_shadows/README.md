# Reference-guided cast-shadow correction

Open `project.json` and choose **Run all**. The two-image replay shows the same
16-bit RGB part under clear lighting and with a sharp shadow at x=104. A small
dark defect and the part boundary occur inside the shadow.

**Detect and compensate reference shadows** uses an aligned unshadowed nominal
reference. It detects large darker regions with similar channel proportions,
then applies one reciprocal-median gain per qualified region. The reference
never replaces the current image pixels. **Measurements** reports gain, fit
support and relative attenuation spread. Inspect **Shadow mask** and **Gain map**
in Results; select **Image** to see corrected pixels.

The independently generated `clear.png` is the expected result for both runs:

- `clear.png`: no shadow regions; the 48-pixel defect remains unchanged.
- `shadow.png`: exactly 9,984 shadow pixels and gain 2. The result equals
  `clear.png` pixel for pixel, including the shadow edge, texture and part outline.
- Defect contrast against the nominal reference remains R/G/B = 8000/7000/6000.
  The minimum shadow area of 256 keeps the small isolated defect out of the
  clear frame's candidate regions.

Open `invalid_fit.json` for a deliberate model failure: joined attenuation
regions of 0.4 and 0.7 violate the spread limit. The block fails and emits
diagnostic masks/data, with no corrected Image or Gain map event.

Run the examples in Debug or Lean; correction and validity are identical.
Graphics Off skips optional region boxes and labels. No plugin or download is
required. The source filename supplies the current image ticket.

This is a neutral, region-constant illumination hypothesis. A large similarly
dark material change can look like a shadow. Use the optional **Fit mask** to
restrict gain estimation to trusted unchanged material, and validate on your
job. Chromatic shadows, penumbrae, camera motion and nonlinear exposure are not
qualified by this fixture. See [model, parameters and limits](../../../docs/reference-shadows.md).

Images, generator, analytic truth and projects are original work under the
repository MIT license. Both projects carry this attribution in a layout comment.
Regenerate with `python "examples/custom tools/reference_shadows/generate.py"`.
