# SimpleITK Tools

Medical-image and volume examples for the separate **SimpleITK** toolbox group.
Install `python -m pip install -e ".[simpleitk]"` in the application's environment.
Open any project normally; no additional Python plugin file is needed.

The shared assets contain **three public MRI cases and their expert segmentations**
from Medical Segmentation Decathlon Task04 Hippocampus. Images/labels are original,
unaltered NIfTI files, licensed **CC BY-SA 4.0**. Code/project authoring is MIT;
the dataset assets retain their own license. [Attribution and source hashes](assets/msd_hippocampus/README.md).

| Folder | Projects and review points |
| --- | --- |
| [metadata_roundtrip](metadata_roundtrip/README.md) | Window an MRI volume, process a slice in OpenCV, remerge tags and inspect recovered physical metadata |
| [geometry_roundtrip](geometry_roundtrip/README.md) | Crop/resize an MRI slice and reconstruct the correct physical origin, spacing and direction |
| [label_roundtrip](label_roundtrip/README.md) | Process one expert-label slice, reinsert it, then compare against the original labels using Dice/Jaccard |

Each folder has projects for cases **001**, **003** and **004**, plus independent
expected measurements. Start with `hippocampus_001.json`. The slice index was chosen
from the supplied labels to show the largest labeled cross-section. Every project
includes an attribution comment.

These are software interoperability/measurement demonstrations. The label example
uses supplied ground-truth labels as an input; it does not claim to segment a new
MRI scan or establish clinical performance.

From the repository root, run a case with:

```powershell
.\.venv\Scripts\opencv-dataflow-run.exe "examples/SimpleITK Tools/metadata_roundtrip/hippocampus_001.json"
```

Use `opencv-dataflow-api run <project> --profile debug --report <new-file.json>
--artifacts <new-directory> --summary` to collect detailed receipts and image review
previews. Native volumes are typed objects; **SimpleITK to OpenCV** provides a
reviewable 2D image without treating Z slices as color channels.

[`build_examples.py`](build_examples.py) regenerates nine projects and their
expectations from the pinned dataset subset. [SimpleITK contracts](../../docs/simpleitk.md)
and [medical roadmap](../../docs/simpleitk-roadmap.md).
