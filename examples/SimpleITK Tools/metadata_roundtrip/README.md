# Metadata through OpenCV

Open [hippocampus_001.json](hippocampus_001.json),
[hippocampus_003.json](hippocampus_003.json) or
[hippocampus_004.json](hippocampus_004.json), then **Run once**.

1. Read the MRI and its original native metadata.
2. Apply an explicit intensity window to uint8, then extract a physical Z slice.
3. Run ordinary OpenCV Gaussian blur, retaining its bridge payload.
4. Reconstruct the native singleton volume. The converter's explicit metadata
   output is wired as well, demonstrating a remerge path.
5. Merge the original tag dictionary and add `ProcessingNote`. The geometry switch
   remains off so the original full-volume geometry does not overwrite slice geometry.
6. Select **Read medical metadata** to inspect recovered XYZ spacing/origin/direction
   and tags; select the final converter to view the processed image.

[`expected.json`](expected.json) records the original size, chosen slice, physical
plane origin and expected custom note. The source tags survive filtering and
snapshots; the uint8 window/blur deliberately changes samples. This is not a
bit-exact MRI intensity round trip.

Images and expert labels: [MSD Task04, CC BY-SA 4.0](../assets/msd_hippocampus/README.md).
[All SimpleITK examples](../README.md).
