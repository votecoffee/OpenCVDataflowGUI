# SimpleReferenceInspect

This project compares parts with a statistical reference trained from eight
generated good images. Normal variation in one small feature is learned.
A separate stable-feature mask is used to estimate exposure correction and
check reference agreement.

## Setup

Install SimpleReferenceInspect into the environment running DataflowGUI.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps "C:\path\to\ReferenceInspect"
```

Open [project.json](project.json) and run once per frame.

| Frame | Result | Defect area |
| --- | --- | ---: |
| nominal.png | Pass | 0 |
| changed.png | Fail | 754 reference pixels |

Both frames use different exposure settings from the good training set.
They are already geometrically aligned. For moving parts, supply an affine
pose from the recipe reference grid into the current image.

## Train and review

1. Select the inspection block and open **Train reference inspection…**.
   The recipe restores all eight good samples and both masks.
2. Review **Reference mean**, **Normal variation**, **Inspection mask**,
   and **Normalization mask** in the view selector.
3. Remove a sample from the filmstrip or edit a threshold, then select
   **Train references**. The training health display reports the remaining
   samples and learned variation.
4. Open `assets/nominal.png` or `assets/changed.png` with **Test image…**,
   then select **Inspect test / selected image**. Review the overlay and
   deviation map. Scroll to zoom and drag to pan.
5. **Stage recipe** proposes a new trained recipe. **Apply** commits it to
   the block. Save the graph to retain the revision.

The popup takes aligned images. The inspection and normalization masks
can be imported from nonzero grayscale mask images.

## Outputs

The deviation map is the residual magnitude divided by the learned
tolerance. Values above 1 exceed that tolerance. The signed residual
identifies brighter and darker changes. The defect mask includes connected
regions meeting the configured minimum area. Raster outputs and the overlay
use source-image coordinates. Areas and reported bounds use the reference
grid.

Check `inspection.evaluated` along with the verdict. A failed coverage,
exposure, or reference-agreement gate prevents defect evaluation and leaves
the inspected mask empty.

## Data

All sample images, masks, and geometry were generated for this demonstration.
The recipe retains the aligned training images. No external image datasets
are included.
