# Crop/resize with physical coordinates

Open [hippocampus_001.json](hippocampus_001.json),
[hippocampus_003.json](hippocampus_003.json) or
[hippocampus_004.json](hippocampus_004.json), then **Run once**.

A native MRI slice passes through ordinary OpenCV crop and 2× resize. **OpenCV to
SimpleITK** composes their pixel-center transforms and reconstructs a 3D singleton
volume. Its spacing becomes `[0.5, 0.5, 1.0]` in this dataset's physical coordinate
units. The crop starts at `(2,3)` and 2× resize maps output center `(0,0)` to original
center `(1.75,2.75)`; the origin follows the native source direction matrix.

Inspect **Read medical metadata** and compare with [`expected.json`](expected.json).
The final Bone colormap is a display conversion; it does not change the medical
image used for reconstruction. No image is automatically resampled back to the
full volume. Reinsertion requires the original full slice grid and rejects this
cropped/resized output.

Images and expert labels: [MSD Task04, CC BY-SA 4.0](../assets/msd_hippocampus/README.md).
[All SimpleITK examples](../README.md).
