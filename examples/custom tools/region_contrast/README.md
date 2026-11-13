# Signal/background contrast

Open `project.json` and **Run all**. The first original 16-bit RGB image,
`strong.png`, must pass; `weak.png` must fail the red-channel CNR limit of 5.
Their expected values are **7.0710678119** and **2.8284271247**. The weak image's
failure is intentional and should appear in the filmstrip and ticket results.
The headless runner therefore returns a nonzero exit code for the full set.

Each editable rectangle covers 1,280 pixels with balanced +/-400 checker
variation. Both population standard deviations are exactly 400 counts.
Signal means differ from the 20,000-count background by 4,000 or 1,600 red
counts. `assets/truth.json` derives expected ratios directly from those values.
The report includes separate R/G/B measurements; the limit uses **R only**.
Change the selected channel or choose Gray to explore another criterion.

**Measure signal/background contrast** is under **Custom > Inspection > Visibility**.
Signal outlines are cyan and background outlines orange. The input samples
remain unchanged; result graphics are objects overlaid for display. Use the
region editors to try circles/polygons, or expose the mask inputs for irregular
selections. A work mask/region can restrict both selections.

Open `invalid_selection.json` and **Run once** for an intentional overlap error.
It must report `overlapping_selections`, emit diagnostic masks/counts, and
withhold **Value** and measurement scalars. An empty or insufficient selection
is also invalid. Overlap policies can instead exclude signal pixels from the
background or deliberately allow shared samples.

Zero measured noise leaves CNR undefined unless an explicit positive raw-unit
noise floor is configured. Signed data can still produce valid differences/CNR;
the normalized intensity contrasts require nonnegative samples and positive
denominators. Optional threshold/polarity limits assign a tool result. They do
not assign a final ticket judgment or estimate a probability of detection.

Images, graph, generator and analytic reference data are original content under
the repository's MIT license. Regenerate with
`python "examples/custom tools/region_contrast/generate.py"`.
See [contrast definitions and limits](../../../docs/region-contrast.md) and
[OpenCV array statistics](https://docs.opencv.org/5.0/main_modules/core_array.html).
