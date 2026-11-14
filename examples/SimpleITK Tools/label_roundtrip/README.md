# Process labels and measure the deliberate change

Open [hippocampus_001.json](hippocampus_001.json),
[hippocampus_003.json](hippocampus_003.json) or
[hippocampus_004.json](hippocampus_004.json), then **Run once**.

This graph starts with the **provided expert segmentation**, extracts its largest
labeled Z cross-section, applies OpenCV median filtering, reconstructs the slice
and reinserts it into the original volume. All other label voxels remain unchanged.
**Medical label overlap / Dice** compares label 1 in this deliberately modified
volume with the original expert labels, with a minimum Dice of 0.9.

Select the overlap block for Dice, Jaccard and voxel counts. The parallel MRI
window/converter shows the matching source anatomy. The label display uses a fixed
0–2 Viridis palette: zero is background, one anterior and two posterior hippocampus.

[`expected.json`](expected.json) gives independently counted intersections/unions,
changed voxels and expected Dice/Jaccard for all three cases. The median operation
is the same native OpenCV operation; the expected overlap arithmetic is independent
of SimpleITK's metric filter.

This validates volume/slice interoperability and metrics. It is not a trained
MRI segmentation example: the supplied expert labels are already an input.

Images and expert labels: [MSD Task04, CC BY-SA 4.0](../assets/msd_hippocampus/README.md).
[All SimpleITK examples](../README.md).
