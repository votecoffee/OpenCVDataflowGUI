# SimpleITK medical images

Install the optional extra in the environment running the application:

```powershell
python -m pip install -e ".[simpleitk]"
```

The verified development environment uses **SimpleITK 2.5.7**, Python 3.14.3 and
OpenCV 5.0.0. Tools appear under the separate **SimpleITK** toolbox root, organized
as I/O, Converters, Metadata, Processing, Segmentation and Measurements. Existing
**scikit-image** and **Simple Plugins** roots remain separate. Without SimpleITK,
the catalog and ordinary OpenCV tools remain usable; a medical execution explains
which extra to install.

## Initial tools

| Block ID | Purpose |
| --- | --- |
| `input.simpleitk` | Read a single 2D image or 3D volume using native medical I/O |
| `output.simpleitk` | Exclusively create a new `.nii`, `.nii.gz`, `.mha` or `.nrrd` file |
| `sitk.to_opencv` | Extract a 2D scalar/color image or a selected physical volume slice; expose bridge metadata |
| `sitk.from_opencv` | Reconstruct a medical image using attached or explicitly wired bridge metadata |
| `sitk.read_metadata` | Expose geometry, pixel description and string tags as editable JSON data |
| `sitk.apply_metadata` | Merge/update/remove tags, optionally assign explicitly supplied geometry |
| `sitk.insert_slice` | Reinsert a processed full slice into its original 3D reference volume |
| `sitk.window` | Explicit raw-intensity window and chosen output sample type |
| `sitk.gaussian` | Recursive Gaussian smoothing with sigma in physical units |
| `sitk.threshold` | Inclusive scalar binary threshold over 2D/3D data |
| `sitk.otsu` | Native Otsu with bins, output labels, midpoint option and optional physical mask |
| `sitk.label_overlap` | Selected-label Dice/Jaccard and counts on matching physical grids |

See the generated [parameter reference](block-parameters.md) or
`opencv-dataflow-api describe sitk.to_opencv` for current controls, limits and enum
values. New tools use the normal block executor: tickets, outcomes, validation,
exceptions, timing, logging, nested graphs, previews for processing blocks and
saved JSON/XML all follow the shared contracts.

## Physical images and array order

A medical `sitk` connection carries a `MedicalImage` with its native SimpleITK
image, rather than pretending a volume is an OpenCV color array. Direct medical
→ OpenCV image connections are rejected: use the converters. Arbitrary vector
components are not automatically interpreted as RGB.

SimpleITK geometry uses **XYZ** index/size order; NumPy arrays use **ZYX** for
volumes, and YX for 2D planes. Physical points follow
`origin + direction @ (spacing * index)`. The direction columns may be oblique or
reflected. SimpleITK does not attach an intrinsic unit name to origin/spacing;
keep the source's units consistent and retain its metadata. File-reader coordinate
conventions remain native SimpleITK conventions; no extra RAS/LPS conversion is
silently applied. [Official image concepts](https://simpleitk.readthedocs.io/en/release/fundamentalConcepts.html),
[array order and image details](https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/03_Image_Details.html).

The current boundary supports 2D/3D real scalar/vector images, up to 16 components,
32 million samples and 256 MiB per image. Complex/RLE-label/4D images need explicit
conversion or future tools. Integer and floating sample types stay unchanged in
the bridge. Individual OpenCV blocks retain their own dtype/channel requirements;
window/cast explicitly where needed. RGB/RGBA vectors require an explicit mode,
or select one component of a general vector. A 3D extraction selects X, Y or Z
and a zero-based slice; the remaining axes retain their original ordering.

## Carry, expose and remerge metadata

`SimpleITK to OpenCV` emits an ordinary tagged image plus a JSON `metadata` output.
The image's connection payload also carries `simpleitk_bridge_v1`:

- Original dimension, XYZ size, spacing, origin, direction, pixel description and
  the complete bounded string-tag dictionary (`simpleitk_metadata_v1`).
- Source identity, selected axis/index/component and an independent root plane frame.
- The normal `spatial.to_root` mapping, updated by supported OpenCV geometry tools.

Ordinary OpenCV processing retains the payload automatically. Wire the image
through processing to **OpenCV to SimpleITK**. Its optional metadata input can also
receive the converter's explicit metadata output. **Read medical metadata** exposes
native geometry/tags, and **Apply medical metadata** can merge a captured dictionary
and user-entered tag changes. Geometry assignment is off by default; enabling it
requires matching actual dimensions and does not resample pixels.

For crop/resize/pyramid/affine paths, reconstruction maps current pixel centers
back to the source plane and recomputes the physical origin, direction and spacing.
It accepts only affine mappings with orthogonal physical axes. A projective warp
or physical shear fails clearly because it cannot be represented by the supported
orthonormal direction/spacing contract. Different source planes also fail. Custom
blocks that move pixels must provide correct spatial provenance.

A 3D slice reconstructs as a **3D singleton volume**, keeping its true oblique
position and slice thickness. Dropping to a 2D physical image would discard that
plane's position/orientation in 3D. **Reinsert processed medical slice** replaces
only that slice in the original volume. It requires the same source identity,
full slice grid, sample type and physical geometry; cropped/resized slices need
an explicit resampling step and are rejected by this first reinsertion tool.

When a custom path loses spatial provenance, reconstruction rejects it by default.
The **Assume same grid** option permits deliberate reattachment only when the
original plane dimensions are unchanged. It is an explicit caller assertion;
dimensions alone cannot prove that a custom operation kept the pixel mapping.

Native snapshots are detached and dictionary tags copied explicitly. The bridge
does not rely on Python `deepcopy(SimpleITK.Image)` or `CopyInformation` to retain
tags; `CopyInformation` omits the dictionary.
[Official image API](https://simpleitk.org/doxygen/v2_5/html/classitk_1_1simple_1_1Image.html).
Runtime metadata and snapshots are separate from project configuration. To retain
them outside a run, export the metadata output/receipt deliberately. Medical file
formats have their own dictionary restrictions: NIfTI does not preserve arbitrary
tags, and the tested native MHA/NRRD readers rename colon-delimited keys. In-memory
tag preservation and native pixel/physical-grid file round trips are tested
separately. Lossless bound metadata sidecars are a roadmap item.

## Ground-truth examples and measurement scope

[SimpleITK Tools](<../examples/SimpleITK Tools/README.md>) contains three original,
unaltered MRI volumes with expert labels from Medical Segmentation Decathlon
Task04 Hippocampus, under **CC BY-SA 4.0**. Nine projects demonstrate metadata
remapping, crop/resize geometry and processed-label reinsertion/overlap. Every
project contains attribution and source hashes are recorded alongside the assets.
[Dataset and license record](https://registry.opendata.aws/msd/).

The label example starts from the supplied ground-truth labels and changes one
slice. Its Dice values measure those deliberate software changes; they do not
evaluate an MRI segmentation model. Missing labels in both inputs are reported
as undefined, rather than assigned a perfect score. Prediction/reference grids
must match; explicitly tolerated tiny geometry differences are treated as rounding
before the native comparison. There is no implicit registration/resampling.

The [medical roadmap](simpleitk-roadmap.md) separates this foundation from future
DICOM-series handling, volumetric segmentation, registration, resampling and
volume review.
