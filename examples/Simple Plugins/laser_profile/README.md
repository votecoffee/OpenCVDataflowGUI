# SimpleLaserProfile examples

Open a project below with **File → Open**, then use **Run once**. The app loads
the required Simple Plugin automatically. Each listed folder contains runnable
project JSON files directly; testing and development material is in `dev`.

| Example folder | Start here |
| --- | --- |
| [inspection](<inspection/README.md>) | [calibrated_step.json](<inspection/calibrated_step.json>) |
| [training](<training/README.md>) | [calibrated_step_trained.json](<training/calibrated_step_trained.json>), [calibrated_unavailable_trained.json](<training/calibrated_unavailable_trained.json>) |

Calibrated step cases are generated metric controls. Real WLI images remain
pixel-only; missing observations and availability failures are retained.

Use the block's Configuration popup to inspect or edit retained training. In
`training`, prefer a `*_trained.json` project when you want the finished staged setup.

[Detailed setup, sources and limitations](dev/README.md) ·
[Development and verification files](dev/)
