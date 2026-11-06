# Crowded tray grid

After [loading the trusted matcher plugin](../README.md), open
[inspection.json](inspection.json) and run all four frames. Each tray has
seven columns and six rows of the same asymmetric part. The scenes vary
position, angle, scale, contrast, and background noise.

The pinned recipe should accept **42 parts in each frame**. Inspect
**Matches** for the source-image pose of every part. The cyan axes mark the
trained datum in the fit overlay. [Expected counts and datums](expected.json)
