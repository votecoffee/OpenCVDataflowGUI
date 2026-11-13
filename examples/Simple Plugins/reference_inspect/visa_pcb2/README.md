# SimpleReferenceInspect · posed real VisA PCB2

Open [project.json](project.json) with SimplePatternMatch and
**SimpleReferenceInspect dev3 or later** installed. Run eight times.
The matcher supplies the current PCB pose; reference inspection compares its
appearance with eight good, pose-aligned training images. Reopening the trainer
restores all eight good images, their support, the fixture and both masks.

This is a retained **hard case**, with real normal/anomalous images and
publisher-provided defect masks. It exposes current errors:

- All four independent normal photos currently raise false alarms.
- All four anomalous images are flagged, but some flags come from unrelated
  differences. One annotated defect has zero pixel recall.
- Pose and exposure gates pass. Default inspection limits were retained;
  they were not loosened to make these test images pass.

[evaluation.json](evaluation.json) includes every result, defect-mask precision,
recall and covered ground truth. The native result maps and assessment pages
are in the results folders. The example is not qualified PCB inspection.
Its purpose is to make the alignment/training workflow runnable and its current
accuracy limits visible.

## Split and training workflow

The publisher's pinned one-class split is preserved: select the first eight
training-normal filenames, first four test-normal filenames and first four
test-anomaly filenames in sorted order. No test image teaches the model.
The same fixed board inspection and stable-feature masks are used throughout.
The inspection mask includes the upper leads as well as the board face.

The existing PCB matcher supplies all training/runtime poses. No ground-truth
pose or defect mask is connected to the graph. Defect masks are used only to
evaluate the held-out results. The matcher recipe remains unchanged from the
existing real-PCB localization example.

Run a frame, open **Train reference inspection…**, and choose
**Inspect wired image + pose** followed by **Review last inspection…**.
Review differences before changing the good training set. Keep the held-out
images out of training; use an additional validation split for adjustments.

## Rebuild

From the host repository root, download the selected pinned sources and build
into a new directory:

```powershell
python "examples/Simple Plugins/reference_inspect/visa_pcb2/download_sources.py" .local/visa-pcb2-source
python "examples/Simple Plugins/reference_inspect/visa_pcb2/build_example.py" .local/visa-pcb2-source .local/new-visa-reference
```

The source cache needs 16 real photographs and four original defect masks.
The portable output includes resized photographs, evaluation-only masks and
the retained trained model. [Attribution, hashes and changes](SOURCES.md).
