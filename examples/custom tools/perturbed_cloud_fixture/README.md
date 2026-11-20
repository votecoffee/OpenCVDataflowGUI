# Perturbed point-cloud fixture

Open `perturbed_cloud_fixture.json` and run the graph. The fixture starts
with a 192-point two-surface reference in millimeters. Seed 7631 draws a
known rigid pose, drops 38 rows, and adds independent Gaussian noise in
the destination XYZ axes. The block emits the clean transformed scan,
perturbed scan, exact source-to-target pose, and source-to-output row pairs.

- **Recover the known rigid pose** fits the 154 available pairs. Its pose
  can be compared directly with the fixture's ground-truth pose.
- **Sample the perturbed scan** keeps 24 rows. Source-index attributes
  trace each sample back to the original reference.
- **0.8 mm tolerance** accepts the synthetic noise. **0.08 mm tolerance**
  intentionally fails and emits the out-of-tolerance points. The headless
  project therefore returns a failed inspection.

Pose, noise and dropout use independent PCG64 streams. Changing dropout
keeps the noise draw for each surviving source row. Fixed mode accepts an
axis-angle vector in degrees and XYZ translation in the cloud's units.
Seeded mode draws a uniform axis and a bounded angle and translation.
Noise scales are Gaussian standard deviations or uniform half-widths.
RGB, rotated normals and aligned row attributes remain available.
Retained source pixels refer to the original capture. Noise statistics
measure the differences in the emitted float32 clouds.

The generator admits up to 4,096 source rows so its pairs can feed the
matched-transform block directly. The synthetic fixture uses a local
seeded equivalent of the OpenCV perturbation helpers. The reference PLY
is original synthetic geometry by OpenCVDataflowGUI contributors under
the repository's MIT license. Run `build_example.py --output <empty
directory>` to regenerate the asset and project.
