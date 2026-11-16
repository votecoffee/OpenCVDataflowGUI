# SimpleSurfaceEnhance map reviews

Open one of the projects below in OpenCVDataflowGUI. These file-only reviews
need no plugin, package installation, model download or hardware connection.
Each project loads eight retained RGB views. Select a node to inspect its
source-aligned map, or select Overview for the combined fit/coverage legend.
No estimator runs when opening or validating these projects.

This E1 delivery includes a standalone **0.1.0.dev2** numerical tool for replay.
Its retained training editor and execution plugin are the next milestone.
Normal vectors are orientation evidence; no measured height or XYZ point cloud
is supplied. See [validation](VALIDATION.md) for the checks and scope.

Five controls use the same generic estimator settings. Each folder has its
capture manifest, original numerical inputs, rendering-reference normals/albedo,
raw result maps, evaluation and image views. `project.json` is an optional
OpenCVDataflowGUI **file-only review**; file-source PASS means loading succeeded,
and does not execute surface estimation.

| Case | Valid coverage | Mean rendering-normal error | What to assess |
| --- | ---: | ---: | --- |
| [Analytic clean](studies/analytic_clean/project.json) | 100% | 0.059° | Bump/hollow geometry separated from varying albedo. |
| [Analytic contaminated](studies/analytic_contaminated/project.json) | 98.97% | 0.059° | Highlight and shadow observations excluded; 200 missing/ceiling pixels unavailable. |
| [Four-light ambiguity](studies/four_light_ambiguity/project.json) | 61.79% | 0.071° on available pixels | The 7,392 competing-normal pixels remain unavailable. |
| [Concrete material clean](studies/polyhaven_clean/project.json) | 100% | 0.062° | Normal/albedo reference maps from a public CC0 material. |
| [Concrete material contaminated](studies/polyhaven_contaminated/project.json) | 99.69% | 0.062° | 26,709 observations rejected; supported normals retained. |

These errors compare against the inputs used to render the light stacks.
They are **software controls**, not physical normal/height accuracy results.
The public material is **Anti Slip Concrete by Charlotte Baglioni, Poly Haven**,
under CC0. **Powered by Poly Haven.** Full attribution, hashes and transformation
details are in [DATASET_CREDITS.md](DATASET_CREDITS.md).

The concrete material's linear EXR green channel supplies relative albedo.
Its unsigned OpenGL normal PNG is decoded, area-averaged to 256×256 and
renormalized, then rotated `[Nx,-Ny,-Nz]` into camera axes. Local diffuse
cosine shading, fixed light gains/exposures and deterministic noise generate
the test frames. The same normal/albedo arrays are retained as rendering
references. No actual material BRDF, cast shadows or multi-light camera
measurement is claimed by this construction.

The analytic reference is independent of the solver: derivatives of a known
bump/hollow define its normals, with a varying albedo pattern. Both families
use identical fit settings; only the four-light ambiguity control changes
the declared hardware's light count. No result-driven threshold tuning.

## Replay estimation

The included wheel is explicitly selected for this command; existing host
dependencies are reused without installing into its environment. From this
example folder, use the Python executable that runs OpenCVDataflowGUI:

```powershell
python run_capture.py --wheel releases/opencv_simple_surface_enhance-0.1.0.dev2-py3-none-any.whl studies/polyhaven_contaminated/capture.json --output new-review
```

Use a new output directory. Add `--no-graphics` for identical raw numerical maps
without PNG rendering. `maps.npz` retains 14 raw maps, unavailable NaN pixels,
normal/albedo/support and every observation's fit evidence. `report.json`
records calibration, units, coverage, reasons and display scales. Coverage is
availability, not probability or an accuracy score. `normal_change` measures
vector change per source pixel, not metric curvature.

`verify_package.py` replays all five controls through the explicitly selected
wheel, compares maps against retained results and writes a new verification
receipt. Its optional `--source` check is for package maintainers.

## Rebuild the controls

`fetch_polyhaven.py` fetches the pinned CC0 material maps using the API's required
identifying User-Agent and verifies publisher hashes. `build_examples.py` renders
the five controls into a new folder; rebuilding additionally needs OpenEXR
3.3..<4 for curation and the surface package on Python's import path.
`build_review_projects.py` adds the file-only projects through the host API.
Runtime estimation/review never downloads data. Original inputs, exact hashes,
transformations and attribution are retained alongside the studies.

## License

Original example/tool code uses [MIT](LICENSE). The **Anti Slip Concrete**
material maps by **Charlotte Baglioni / Poly Haven** are **CC0**.
**Powered by Poly Haven.** See [dataset credits](DATASET_CREDITS.md) for upstream
asset/license links, API terms and dependency notices. No endorsement is implied.
