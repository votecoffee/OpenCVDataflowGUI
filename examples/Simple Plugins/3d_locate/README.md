# Simple3DLocate

For a **real single-view depth capture**, open [T-LESS](real_tless/README.md).
It uses attributed CC BY 4.0 industrial RGB-D data and requires plugin dev3.
This older generated example deliberately uses **complete/fused observations**;
it does not simulate first-surface depth-camera sensing.

This generated example locates two rigid surfaces with independent point
sampling, rotation/translation, partial occlusion, measurement noise, and
unrelated clutter. All coordinates are in millimeters.

For STL CAD training, retained-mesh resampling and bounded candidate search,
open the [CAD surface example](cad_surface/README.md), which requires
Simple3DLocate 0.1.0.dev1 or later.

## Setup and run

Install the Simple3DLocate wheel or source folder into the Python environment
running DataflowGUI, then restart the application.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps "C:\path\to\Simple3DLocate.whl"
```

Open [project.json](project.json) and run once. Expect two accepted instances.
Select Simple3DLocate and inspect its overlay in the result filmstrip or its
`fit` cloud in the 3D viewer. Green model samples have geometric support, red
samples are unsupported or occluded, and gray points belong to the scene.
RGB axes mark each taught model origin and orientation.

Open **Train 3D locator…** to see the retained reference and trained samples.
Load `assets/scene.ply` as the independent test scene and locate it.
To teach a replacement, load `assets/model.ply`, train, test, stage, and Apply.
The model cloud should contain one isolated part.

With plugin **0.1.0.dev8**, run in Debug and use **Capture last processed
inputs** to review the scene and any camera/depth consumed in one completed
run. The trained model is preserved. Acquisition pairing remains explicit;
see [completed-input capture](CAPTURE.md). Older hosts keep manual captures.

## Outputs

The saved model frame is `part`; the scene frame is `camera`.
Poses map model coordinates into scene coordinates using 4×4 rigid transforms.
Support is a fraction of model samples. Point-to-plane RMSE is in millimeters.
Neither measurement is a class probability.

[assets/truth.json](assets/truth.json) contains the generated true transforms.
[expected.json](expected.json) records the locally validated result.
[result.png](result.png) is a computational render of that result.
Rebuild the example with `build_demo.py . --host-project` when the tool and
host Python packages are available.

## Provenance and limitations

The analytic surface generator and these generated clouds use the MIT license
of this repository.

This is a synthetic workflow check. Real depth-camera and dense bin-picking
qualification remain future work. The current OpenCV surface-matching backend
has a [published patent notice](https://docs.opencv.org/4.13.0/d9/d25/group__surface__matching.html).
The tool's dependency notes record it separately from project copyright terms.
