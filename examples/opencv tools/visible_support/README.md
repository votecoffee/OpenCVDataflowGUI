# Candidate support before pick selection

Open a project below and run once. The native template matcher finds three
candidates; **Filter candidates by visible support** checks their expected pixels,
then **Select pick candidate** checks clearance and prefers the downstream part.
The filmstrip overlays green/red candidate outlines and support percentages on
the current image. No camera, optional plugin or download is required.

| Project | Supported samples per part | Expected pick origin | Outcome |
| --- | --- | --- | --- |
| [Clear](clear.json) | 616, 616, 616 | [215,120] | PASS: downstream clear part. |
| [Touching](touching.json) | 616, 616, 616 | [35,28] | PASS: both touching downstream parts fail clearance. |
| [Covered](covered.json) | 616, 616, 406 | [125,70] | PASS: partly covered downstream part is filtered out. |
| [Require three](required_three.json) | 616, 616, 406 | None | Intentional failed inspection: requires three supported parts; pick inputs reject the failure. |

The full template is 31×21 pixels. A region excludes its 7×5 background hole,
leaving **616 expected samples**. Covering ten columns removes 210 expected
samples, leaving **406/616 = 65.91%** appearance support. The same candidate's
native detector score remains about 0.546 and is not overwritten. Touching parts
still have 100% appearance support: clearance is a separate check, not inferred
from the fraction. Coordinates/counts are authored independently in `truth.json`.

Inspect Support's Candidates to see all records, including rejected ones. Matches
contains only accepted records. Try changing tolerance or the minimum fractions;
Absolute tolerance is zero in these exact controls. In `required_three`, the
support outputs and filmstrip remain current despite the failed count gate.
The pick block rejects those failed inputs. Lean keeps counts and pick behavior
while omitting diagnostic drawings.

```powershell
.\.venv\Scripts\opencv-dataflow-run "examples/opencv tools/visible_support/covered.json"
.\.venv\Scripts\python "examples/opencv tools/visible_support/build_example.py"
```

## Attribution and scope

All images, generator, graphs and independent sample/pick truth are original work
by OpenCVDataflowGUI contributors under the repository MIT license. No customer
or third-party images are included. Each graph has an attribution comment.
The seeded textured rectangles are controlled tool fixtures, not real parts or a
trained production inspection. Five PNG hashes, exact sample counts, detector
positions, accepted counts and pick origins are checked. Tests regenerate all
images/projects in a fresh folder and compare exact bytes.

This is a template-appearance visibility proxy. Lighting, pose errors, defects
and lookalike occluders can confound it. It does not establish true occlusion,
grasp safety or robot collision clearance.
[Sampling, transform and metadata contracts](../../../docs/visible-support.md) ·
[All OpenCV examples](../README.md)
