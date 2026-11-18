# SimpleGDRNPP examples

Open a project below with **File → Open**, then use **Run once**. The app loads
the required Simple Plugin automatically. Each listed folder contains runnable
project JSON files directly; testing and development material is in `dev`.

| Example folder | Start here |
| --- | --- |
| [accepted_parts](<accepted_parts/README.md>) | [review_project.json](<accepted_parts/review_project.json>) |
| [live_review](<live_review/README.md>) | [review_project.json](<live_review/review_project.json>) |
| [rejected_part](<rejected_part/README.md>) | [review_project.json](<rejected_part/review_project.json>) |
| [setup](<setup/README.md>) | [project.json](<setup/project.json>) |

**Inference setup:** runtime projects require their configured external worker and
model weights. Start with a `review_project.json` for immediate file-only results.
A retained review or loaded recipe does not establish model or physical accuracy.

Use the block's Configuration popup to inspect or edit retained training. In
`training`, prefer a `*_trained.json` project when you want the finished staged setup.

[Detailed setup, sources and limitations](dev/README.md) ·
[Development and verification files](dev/)
