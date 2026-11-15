# Camera health from a saved reference

Monitor a fixed camera's lighting, position and focus on a real **VisA PCB**
photograph. Ten offline projects share one frozen three-capture reference, two
lighting regions, two focus regions and two optional trained fiducials. New
synthetic parts are excluded before the background is judged.

Open a project below and click **Run**. Select the monitor's Results tab to inspect
the Report, colored region overlays and optional foreground/background masks.
No plugin, hardware connection or optional package is required.

| Project | Expected state | Controlled change |
| --- | --- | --- |
| [Healthy moving part](healthy_moving_part.json) | Healthy / pass | New part in a previously unoccupied location |
| [Position warning](position_warning.json) | Warning / pass | Background translated 2 px horizontally |
| [Position alarm](position_alarm.json) | Alarm / fail | Background translated 5 px horizontally |
| [Lighting warning](lighting_warning.json) | Warning / pass | Global intensity gain 0.85 |
| [Lighting alarm](lighting_alarm.json) | Alarm / fail | Global intensity gain 0.70 |
| [Focus warning](focus_warning.json) | Warning / pass | Gaussian blur, sigma 0.68 px |
| [Focus alarm](focus_alarm.json) | Alarm / fail | Gaussian blur, sigma 1.5 px |
| [Healthy fiducials](fiducials_healthy.json) | Healthy / pass | Two trained static landmarks replace background tracking |
| [Fiducial position alarm](fiducials_alarm.json) | Alarm / fail | Same landmarks translated 5 px |
| [Insufficient evidence](insufficient_evidence.json) | Alarm / fail | Explicit mover/occlusion mask covers the entire image |

Warning/alarm defaults are 1/3 px displacement, 0.25/1 degree rotation,
0.5/2% scale change, 10/20% lighting change and 20/40% focus loss. No limits are
tuned per frame. Lighting uses separate B/G/R means; focus is contrast-normalized
Tenengrad. The original photograph has no physical camera-health labels: all
changes and expected outcomes here are authored software fixtures.

To experiment with training, save a separate project copy, open **Train camera
reference…** and review the retained history and region definitions. Import the
three `assets/training_*.png` captures to train a new reference, then Stage and
Apply. Training and normal execution never update the original pinned reference.
See [camera-health documentation](../../../docs/camera-health.md) for region
editors, threshold units, single-fiducial limitations and unavailable evidence.

Headless replay, from the repository root:

```powershell
python -m opencv_dataflow.runner "examples/custom tools/camera_health/healthy_moving_part.json"
python "examples/custom tools/camera_health/build_example.py"
```

The second command deterministically rebuilds assets/projects from the bundled
source photo. [truth.json](truth.json) records every controlled change and asset
hash. Code is MIT; photographs, edited images and normalized training intensity
arrays retain **CC BY 4.0**. Credit and the upstream hash chain are in
[SOURCES.md](SOURCES.md) and every project's on-canvas Comment.
