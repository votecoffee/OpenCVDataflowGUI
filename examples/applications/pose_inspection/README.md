# Pose-relative inspection

**Data:** controlled synthetic fixture with explicit reference geometry or labels.

Train a geometric pattern and use its detected pose to position a circular inspection region.

Open [`pose_inspection.json`](pose_inspection.json) with **File → Open…**. Run once and inspect the pose-relative region and histogram. Change the pose offset to shift the inspection.

[All examples](../../README.md)


## Controlled fixture

Generated part transforms provide exact pose-relative region truth. See [real PCB localization](<../../Simple Plugins/pattern_matching/visa_pcb2/README.md>) for a photo-based workflow.

New application showcase variants should prefer suitable, attributed real captures.
