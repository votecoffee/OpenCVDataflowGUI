# Retain triangle CAD and inspect first-surface agreement

SimpleBinPicking Jaw `0.1.0.dev10` can retain original-frame triangle geometry
alongside its sampled contact model. **Cloud / mesh_fit** and the trainer's
**Mesh fit** layer show a bounded first-surface diagnostic. Pose acceptance,
contact and clearance continue to use the existing retained surface samples.

## Train and reopen

In **Teach CAD jaw grasps…**, choose **New from bin + triangle CAD…**. Select a
calibrated metric bin/reference recipe, then an STL or triangle PLY. Declare m
or mm coordinates, the original part-frame name and outward triangle winding;
reverse winding explicitly if needed. The trainer preserves the original origin, samples 40,000 surface points
with a fixed seed, and starts the usual editable envelope grasp. Teach opposing
contacts and retract direction, retain an explicit pose/depth reference, preview,
then **Stage → inspector Apply**. Cancel preserves the applied recipe.

The current import route uses the existing Simple3DLocate geometry reader only
while opening a mesh file. The example plugin bundles that reader. Direct
array-based `PartMesh` construction, archive loading and runtime rasterization
need no companion package or additional mandatory dependency.

```python
from opencv_simple_bin_picking import JawRecipe, PartMesh

mesh = PartMesh.read("part.stl", scale=0.001, frame="part")
recipe = JawRecipe(bin_recipe, mesh.sample(), taught_grasps, mesh=mesh)
```

Use matching original part-frame names for mesh, samples and supplied poses.
PartMesh coordinates are meters. File import validates bounded geometry and may
discard degenerate source triangles through the companion reader; the retained
triangles, samples and import-source checksum remain explicit. Reopening restores
the mesh and frozen reference. Edits, undo, preview cancellation and revision
checks use the existing training workflow. Cached review exports include the mesh
in the previewed recipe and its diagnostic counts in graphics metadata.

## Assess the result

Select **Cloud / mesh_fit** for the native colored fly-around thumbnail/popout.
This output isolates predicted first-surface mesh queries. In the trainer, toggle
Mesh fit, Pose fit and Scene to compare them. The existing point view uses four
neighboring depth rays; mesh review uses the same integer pixel as each rendered
ray. Counts and weighting therefore differ and are reported separately.

- Green: supported measured depth at the configured pose tolerance.
- Magenta: predicted mesh in observed free space.
- Blue: observed surface in front of the prediction; the hidden target is unmeasured.
- Orange: missing measured depth.
- Purple: a connected instance label differs from the declared target.

The mesh retains only its nearest outward-facing triangle on a camera ray.
Predicted geometry and holes never supply observed free space. Every component
is marked derived; measured scene context stays separate. Metadata `mesh_fit`
records full counts, residual quantiles, triangle count, ray-grid stride,
availability/reason and **acceptance_applied=false**. Display rows are capped at
256 per state per instance, so their density is not a percentage.

Version dev10 adds explicit diagnostic denominators in `weighting`. **Testable
fraction** is the supported, contradictory or label-mismatched ray population
divided by all expected mesh rays. Foreground occlusion, missing depth and
ambiguous neighborhoods cannot test the predicted surface. **Support/testable**
is support divided by that testable population; it is unavailable when no ray
is testable. A single supported ray can yield 100% conditional support, so always
read it with testable fraction and absolute ray counts. The trainer shows these
two values together. They do not alter acceptance.

The `pixel` and `visible_area` summaries classify exactly the same rays. Pixel
weighting counts camera rays. Visible area estimates surface area by midpoint
quadrature, `z^2 * stride^2 / (fx * fy * abs(dot(normal, ray)))`, with the full
perspective ray and outward unit normal. Silhouettes, partial cells and grazing
incidence can bias this estimate; invalid area weights explicitly make that
summary unavailable. It is not exact visible CAD area, an uncertainty model or
a replacement for pixel counts. Both units and denominators are retained, with
no minimum-coverage or conditional-score acceptance rule implied.

Rasterization uses perspective-correct optical Z. Its grid is at most 640×640;
triangle/pixel work is capped at four million pairs per pose, reducing grid
resolution explicitly when necessary. Camera-plane-crossing meshes abstain;
they are not clipped into invented geometry. Cancellation discards incomplete
review. The NumPy rasterizer is parity-checked with the existing bounded geometry
engine and has independent analytic plane/occlusion controls. Native/GPU renderer
equivalence is not asserted.

## Compatibility and execution

Point-only archives remain version 1. Their saved bytes, numerical reports,
acceptance and graphics-off behavior are preserved. Mesh recipes use version 2
with a checksummed numeric triangle member; older readers reject that version.
Both formats remain bounded to 64 MiB with no pickle or executable code.

Runtime mesh review runs only when review retention/graphics are enabled.
Graphics-off and the report-only CLI skip rasterization. Both return the same
inspection report; a mesh recipe's report declares its diagnostic-only use.
Layer/selection/orbit review consumes cached results and does not rerun planning.

This increment supplies evidence for controlled visibility/weighting studies.
It does not enable automatic pose refinement or replace the current point gate.
Independent sensor/pose qualification, contact proposals, force closure and robot
trajectory qualification remain separate work.
