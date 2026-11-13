# SimpleReferenceInspect result review

Requires **SimpleReferenceInspect 0.1.0.dev3**.
Open [the assessment project](../assessment-project.json) to show the six-view
assessment image in the result filmstrip. The existing generated recipe
and its thresholds are unchanged.

| Case | Assessment |
| --- | --- |
| Independent generated nominal input | [Pass](samples/aligned_0/assessment.png) |
| Generated changed input | [Accepted change](samples/aligned_1/assessment.png) |
| Generated reference outside image support | [Not evaluated](samples/unsupported_reference/assessment.png) |
| Captured fixture reference | [Pass](samples/fixture_capture_0/assessment.png) |
| Translated nominal part | [Pose-aligned pass](samples/fixture_capture_1/assessment.png) |
| Translated changed part | [Pose-aligned change](samples/fixture_capture_2/assessment.png) |
| Real VisA PCB training photo | [Self-reference check](samples/visa_pcb2_0/assessment.png) |
| Separate real VisA normal photo | [Agreement/gain rejection](samples/visa_pcb2_1/assessment.png) |

The [real PCB project](samples/visa_pcb2_project/project.json) runs both
photographs with the assessment enabled. Its reference uses **one** good
image and default settings, with no pose alignment. Both source images have
the dataset label **normal**. The independent normal photo fails prerequisite
gates and is **not evaluated for changes**. This is a useful failure example,
not a trained industrial inspection or a claim that this normal PCB is bad.
The self-reference pass checks plumbing and does not establish accuracy.
Use aligned independent good and changed parts to qualify a real recipe.

Reference/current share the reference grid and one intensity display range.
The input overlay retains input coordinates. Gray residual/deviation pixels
are uninspected; amber coverage marks expected inspection pixels that were
not inspected. Deviation is a tolerance ratio, not a probability, and the
display saturates at two. The actual area, exposure, correlation and coverage
limits are shown separately.

In the trainer, inspect a selected/test image and select **Review last inspection…**.
Scroll or pan the synchronized comparison views, change maps without losing
zoom, and use **100%** or **Fit comparisons**. Other tabs show the input
overlay and assessment. **Export review…** saves PNG views, a page, JSON and
raw reference-grid NPY maps. Review does not inspect again or stage training.

For saved project frames, the builder captures the core result from the
single graph execution rather than reconstructing it from resampled outputs.
The unsupported-pose diagnostic uses an explicit off-image transform with
unchanged recipe settings. [The receipt](samples/receipt.json) records frame
positions, recipe/input/reference hashes and source labels.

From the host repository root, rebuild to a **new** directory:

```powershell
python "examples/Simple Plugins/reference_inspect/result_review/build_views.py" .local/new-reference-review
```

The generated PCB recipe archive has its own new SHA-256 pin; ZIP timestamps
can change archive bytes between builds. Input pixels, reference arrays,
measurement reports and review maps are the comparison evidence.

## Attribution and license

Generated samples and builder code use this repository's MIT license.
VisA photographs, their retained training sample and derived review images
remain **CC BY 4.0**. Authors: Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer, *SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation* (2022).
[Dataset and published license](https://registry.opendata.aws/visa/) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ·
[original member names and preparation](../../pattern_matching/visa_pcb2/SOURCES.md).

This example reuses the host's prepared photographs without changing their
bytes, converts them to grayscale for inspection, and adds normalization,
overlays, maps, resized display panels and annotations. No endorsement is implied.
