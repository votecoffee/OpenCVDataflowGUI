# SimpleReferenceInspect

For real camera images, open the [ARMBench clear-tote example](armbench_tote/README.md)
or the [posed VisA PCB hard case](visa_pcb2/README.md). Both include retained
training, independent replay images, assessment graphics, source licenses and
explicit limitations. The original generated demonstration follows below.

This project compares parts with a statistical reference trained from eight
generated good images. Normal variation in one small feature is learned.
A separate stable-feature mask is used to estimate exposure correction and
check reference agreement.

For a moving part, open the [paired fixture capture example](fixture_capture/README.md).

The [result review gallery](result_review/README.md) adds synchronized comparison
views, an optional assessment image, native-map exports and real VisA PCB gate
diagnostics. Open [assessment-project.json](assessment-project.json) to publish
the assessment in the result filmstrip with the original recipe unchanged.

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
3. Open **Edit inspection mask…** or **Edit normalization mask…** to draw
   or refine a mask on the reference image. Mask editing requires
   SimpleReferenceInspect 0.1.0.dev1 or later.
4. Remove a sample from the filmstrip or edit a threshold, then select
   **Train references**. The training health display reports the remaining
   samples and learned variation.
5. Open `assets/nominal.png` or `assets/changed.png` with **Test image…**,
   then select **Inspect test / selected image**. Review the overlay and
   deviation map. Scroll to zoom and drag to pan.
6. **Stage recipe** proposes a new trained recipe. **Apply** commits it to
   the block. Save the graph to retain the revision.

The popup takes aligned images. The inspection and normalization masks
can be imported from nonzero grayscale mask images.

## Edit masks

In either mask editor, choose a rectangle, circle, or polygon and select
**Draw shape**. Drag a rectangle or circle, or click polygon vertices and
press Enter. Drag handles to refine the shape. Shift-click an edge to add
a polygon point and right-click a point to remove it. Scroll to zoom and
middle-drag or Alt-drag to pan.

Choose **Replace mask**, **Add to mask**, or **Exclude from mask**, then
select **Apply shape**. Repeat to combine regions and holes. **Undo**
reverses edits. **Use mask** accepts the pixels into the trainer, while
**Cancel** keeps the existing training. Retrain after changing a mask.
The saved recipe stores the resulting binary masks.

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

The top-level demonstration uses generated images, masks and geometry, with
aligned training images retained in its recipe. The completed-review and training
subfolders also include real VisA photographs and evaluation masks; their
`SOURCES.md` files preserve CC BY 4.0 attribution and describe modifications.

## Background training and aligned imports

The [dev5 training workflows](training_workflow/README.md) add bounded background
training/cancellation, explicit image-plus-affine import with supported-pixel
review, persistent mask/common-support guidance and saved sample descriptions.
Cancelled, stale or closed jobs cannot publish a model. Existing mask editing/Undo,
independent cached review and Stage/Apply/Cancel remain available.

Load that folder's **pattern_plugin.py**, then **plugin.py**, before opening its
original or trained generated/real-PCB projects. Restart if either tool package is
already loaded. The explicit pinned wheels need no host installation. Both native
workflows pass import/diagnosis/cancel/retrain/review/Stage/Apply/reopen and exact
archive Save As. Unchanged inputs produce exact trained arrays/original health;
nine prior runtime frames retain exact reports/maps/default rasters. The PCB's
existing false alarms and incomplete coverage remain unqualified. See
[attribution and modifications](training_workflow/SOURCES.md).
