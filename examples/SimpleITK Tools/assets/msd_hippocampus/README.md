# Public hippocampus MRI and expert-label subset

Dataset: **Medical Segmentation Decathlon, Task04 Hippocampus**.
Task04 source institution: **Vanderbilt University Medical Center**. Credit the
Medical Segmentation Decathlon contributors, including Simpson A. L., Antonelli M.,
Bakas S., Bilello M., Farahani K., van Ginneken B., and collaborators.

- [Publisher's dataset/license record](https://registry.opendata.aws/msd/)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- [Dataset paper and author citation](https://arxiv.org/abs/1902.09063)
- [Archive used](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar)
- License: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
- [Exact archive/member/file SHA-256 and label meanings](SOURCES.json)

Accessed 2026-11-14. Only training cases **001**, **003** and **004** are included,
each with its MRI and expert annotation. The six `.nii.gz` files are copied
**byte-for-byte**, with no image, orientation, metadata or annotation changes.
Subset selection is the only modification. Total bundled source data is 215,907 bytes.

Labels in the source manifest: `0` background, `1` anterior hippocampus,
`2` posterior hippocampus. These are the supplied expert annotations, not outputs
of an algorithm in this repository. Original NIfTI geometry and tags are preserved
as source bytes; the native reader interprets their physical coordinates.

The repository's MIT license covers its own source/code. These MRI and annotation
assets remain under **CC BY-SA 4.0**, including its attribution and share-alike
conditions for redistributed adaptations. Preserve this attribution and source
record with shared copies or derived example assets.
