# Completed Simple3DMeasure review

Load **plugin.py first** through File → Load Python plugin, then open a project
below. Restart if another measurement version is already imported. The explicit
checksum-pinned dev6 wheel loads without an environment installation. A current
host with paired completed snapshots, independent Results review and Debug input/
output retention is required. Flatness examples require the host's optional SciPy.

| Project | Evidence |
| --- | --- |
| [Real frame 13](../../inspection/real13.json) | Measured T-LESS first returns, supplied fixture, recorded plane/residual evidence. |
| [Real frame 24](../../inspection/real24.json) | A separate capture with the same trained regions and bands. |
| [Known step](../../inspection/good.json) | Independent generated 5 mm height surfaces; bounded noise. |
| [Raised feature](../../inspection/raised.json) | Datum fit excludes a raised patch while its surface-range measurement fails. |
| [Missing surface](../../inspection/missing.json) | Height/angle/range stay unavailable; datum probe stays available. |
| [Nonparallel lines](../../inspection/line_guard.json) | Small line residuals coexist with an unavailable guarded separation. |
| [Minimum-zone flatness](../../inspection/flatness.json) | Known generated finite-point zone geometry, distinct from least-squares residual range. |
| [Runtime graphics off](../../inspection/graphics_off.json) | Same real frame 13/model, no emitted graphics; review derives displays from recorded fits only. |

Run once, select the tool and open **Review completed 3D measurements…** on
Results. Read the full value/units/limit/status/reason table; select a measurement
or region, switch fit/residual/exact-input views, toggle layers and use native
Flyaround. Residual mode shows a single-region physical histogram and tail counts.
Export frozen review keeps original recipe bytes, native arrays, full residuals,
PLY/JSON and readable assessment/histogram PNGs. It never refits or measures.
See [workflow and evidence semantics](COMPLETED_REVIEW.md).

Original models, fits, limits, datum/fixture declarations, point sampling and
workers are unchanged. The graphics-off control changes only rendering flags.
Shared content-addressed assets avoid repeated copies of the large trained
reference between projects. Model files remain byte exact with their original pins.

The real cases use publisher-supplied annotation fixtures, not detected poses.
They have no configured acceptance limits or independent gauge truth. Their passes
mean measurements are available, not qualified dimensional accuracy. The same
nonplanar part regions have substantial residual ranges; no settings were tuned
to improve these views. Generated controls are numerical examples, not acquired
camera or gauge acceptance. Residuals are fit evidence, not uncertainty or
deviation from the taught reference.

`expected.json` retains every numerical output. `native-verification.json` records
all eight completed cases, source/model/cloud hashes, selected evidence and five
self-contained [frozen exports](snapshots). Baseline dev5 comparison, source/host
native review, Save As/reopen and fresh Windows checkout receipts are included.
Hands-on operator and independent metrology qualification remain open.

```powershell
python examples.py verify . path/to/new-verification --native
python examples.py build path/to/public-3d-measure-examples path/to/new-example
```

Run with the host Python from this folder; output directories must be new.
Code/generated fixtures are [MIT](LICENSE). Real-data derivatives retain
[T-LESS CC BY 4.0 attribution](SOURCES.md).
