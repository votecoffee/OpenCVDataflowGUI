# SimplePathWidth · Capture a fixture pose

The [project](project.json) uses three generated grayscale images. A bright
circle acts as an independent fixture mark. **In range mask** isolates
the mark, **Find blobs** locates it, and **Pose from feature** wires that
origin to the inspection block. The bead has its own path and never
determines the fixture position.

Use SimplePathWidth 0.1.0.dev4 or later and a DataflowGUI build with wired
data capture. See the [package setup](../README.md#setup).

1. Open the project and run once in Debug to publish the reference image and
   its fixture pose. Select **Measure bead**, then **Train path width…**.
2. Select **Capture image + fixture**. The displayed fixture origin should be
   x 35, y 49. The saved path is already positioned over the reference bead.
3. Use **Stage recipe**, then **Apply** in the block inspector. Save the
   graph to keep the new recipe and fixture.
4. Run twice more to inspect the moved bead and the bead shifted relative
   to the fixture. Use **Result / overlay** or the image results filmstrip
   to see where the cross-scans landed.

| Frame | Width | Signed center offset | Verdict |
| --- | ---: | ---: | --- |
| Reference | 18.0 px | +0.5 px | Pass |
| Fixture and bead move together | 18.0 px | +0.5 px | Pass |
| Bead moves 4 px farther than the fixture | 18.0 px | +4.5 px | Fail |

The saved project already contains the reference fixture, so replay also
works immediately after opening. The capture workflow requires a matching
image and pose from the same acquisition. Use a stable part feature or
fiducial outside the bead when setting up a production inspection.
