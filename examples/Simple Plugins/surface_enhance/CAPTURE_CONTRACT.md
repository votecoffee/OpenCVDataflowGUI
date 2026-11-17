# Surface capture and retained setup

`SurfaceRecipe` retains a normalized floating N×H×W monochrome source stack,
ordered `LightRig`, fit settings, black/ceiling levels, source-grid/capture IDs,
inspection mask, virtual light and JSON provenance. Arrays/calibration are copied
and read-only. The bounded `.sse` archive uses numeric NPY arrays, strict JSON,
member SHA-256 and a pinned whole-archive digest; no executable code or pickle.
Saving preserves existing filenames. Sealed recipes are copied by host Save As.

The setup imports `simple_surface_capture/v1` manifests, imports a recipe or
copies a connected stack. A manifest's images/mask resolve relative to that
manifest. Each light declares ID, unit direction from surface toward light,
relative power and exposure seconds. `input_scale` converts source code into
0..1 without independent normalization. `response` must explicitly be `linear`.
The [retained controls](studies/analytic_clean/capture.json) demonstrate
the complete file format. Directions/power are known imported calibration;
the tool does not estimate the lighting geometry from unknown images.

The host's `capture` connector uses the existing **any** port with semantic type
`surface_illumination_stack`, so arrays stay numerical instead of expanding into
JSON lists. Its value is a Python dictionary:

```python
capture = {
    "schema": "simple_surface_stack/v1",
    "images": normalized_images,       # float32/float64 N x H x W, bounded
    "lighting": {
        "light_ids": list(rig.light_ids),
        "directions": rig.directions.tolist(),
        "relative_intensity": rig.relative_intensity.tolist(),
        "exposure_s": actual_exposure_seconds,
        "frame": rig.frame,
    },
    "capture_id": "part_002",
    "pixel_grid": "camera_original",
    "response": "linear",
    "black_level": 0.02,
    "input_ceiling": 1.0,
    "mask": optional_source_mask,
}
```

`capture_from_recipe(recipe)` constructs a reference capture with this schema.
`retain_capture(capture)` privately copies it into a new reference recipe.
`execute(recipe, capture, estimator=compiled)` validates the full capture before
estimating. Light order/directions/power/frame, source dimensions/grid and
black/ceiling/response must match the reference. Declared exposure seconds may
change; each observation is normalized using its actual exposure while the
compiled direction schedule is reused. Undeclared exposure changes cannot be
inferred. A runtime mask intersects the retained mask; it cannot expand inspection.

Pixel-grid/capture/editor-copy IDs are caller provenance, not measured proof of
registration, fixed pose or synchronized exposures. Keep the camera/object fixed
through a stack, avoid motion, linearize the actual camera response upstream and
provide known directional lighting. Near-field/spatially varying lights need a
different supported calibration model.

The block's `maps` connector similarly uses an array-preserving **any** port,
semantic type `surface_maps`. Fourteen arrays retain source coordinates and NaN
availability. The `report` is ordinary JSON data. `image` is raw floating virtual
relighting in source coordinates. `overview` and eight Image results views are
optional diagnostic rasters; disabling graphics preserves every numerical map.
Neither map vectors nor editor decorations become metric XYZ/cloud points.
