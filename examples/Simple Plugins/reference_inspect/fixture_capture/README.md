# SimpleReferenceInspect · Paired fixture capture

Use SimpleReferenceInspect 0.1.0.dev2 or later. Open
[project.json](project.json) and run once per frame. The locator follows
a small bright fixture mark outside the inspection region.

| Frame | Fixture translation from the taught image | Verdict | Defect area |
| --- | --- | --- | ---: |
| Nominal | 0, 0 pixels | Pass | 0 |
| Moved nominal | +8, −5 pixels | Pass | 0 |
| Moved material change | −6, +7 pixels | Fail | 754 reference pixels |

The saved recipe contains eight good samples and the fixture pose from
the reference grid. Select **Result / overlay** on the inspection block
to see the result in the current image coordinates.

## Capture and train

1. Run the first or second good frame, then open **Train reference
   inspection…**. The popup restores the good samples and reference pose.
2. Select **Capture image + pose** to append the current good part.
   The capture validates that the image and pose belong to the same
   acquisition and aligns the sample into the taught reference grid.
3. Choose **Common sample coverage** to see the pixels present in every
   good sample. Keep the inspection and normalization masks inside that
   coverage, then select **Train references**.
4. **Inspect wired image + pose** previews the current frame without
   adding a sample. Use this on the third frame to see the material change.
5. Stage, apply, and save the graph to keep the updated recipe.

**New reference set** clears the draft samples and masks. Its first paired
capture teaches a new reference image and pose. The saved block stays at
its previous recipe until staging and applying the new training.

The fixture mark stays separate from the inspected material so its
movement follows the part. The example uses generated images and a known
translation sequence. The pixels exposed by alignment at the frame
boundary are excluded by the taught masks and retained in each sample's
support metadata.
