# Sources, checkpoints and data

The standalone inference topology, crop construction and pose decoding adapt
the Apache-2.0 GDRNPP implementation at commit
`990003d2c13d4a06fec98e27d980595a9831ccc7`. The modified modules identify their
source files and changes. The corresponding license is included in
[`licenses/GDRNPP_Apache-2.0.txt`](licenses/GDRNPP_Apache-2.0.txt).
Other original code uses the [MIT license](LICENSE).

Credit Xingyu Liu, Ruida Zhang, Chenyangguang Zhang, Gu Wang, Jiwen Tang,
Zhigang Li and Xiangyang Ji, **GDRNPP: A Geometry-guided and Fully Learning-based
Object Pose Estimator**, TPAMI 2025; and Gu Wang, Fabian Manhardt, Federico
Tombari and Xiangyang Ji, **GDR-Net**, CVPR 2021.
[Publisher source](https://github.com/THU-DA-6D-Pose-Group/gdrnpp_bop2022).

## Selected model

The author-linked [ModelScope repository](https://modelscope.cn/models/wangg12/GDRNPP)
declares **Apache License 2.0** in its repository metadata. The selected asset is
the PBR-only, single-object T-LESS object-6 ConvNeXt checkpoint. Its published
size is 410,204,745 bytes and SHA-256 is
`b1dc54a1a6572e4db560f28a05a0821ed4dc05b18c07052f651f890753d8e606`.
Those actual bytes were downloaded and verified. The publisher supplies no
separate file-level license document beside that checkpoint; this records the
repository declaration rather than inventing an additional grant. Weights
remain external and are not included in source, wheels or examples.

Source configuration and framework runtime have separate provenance. The
inference adapter retains the selected topology and all 392 state entries,
including normalization aliases. It does not install Detectron2, MMCV, EGL,
training losses or custom CUDA extensions. The reference check executes only
explicitly selected definitions from the audited source; training initializers
are stubbed before strict checkpoint loading. It is not a qualification of the
publisher's complete training/evaluation distribution.

## Runtime dependencies

- [timm 0.6.7](https://github.com/huggingface/pytorch-image-models/blob/v0.6.7/LICENSE):
  Apache-2.0, matching the upstream requirement. ConvNeXt is created without
  automatically fetching other pretrained weights.
- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) and
  [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE): BSD-style
  source terms; installed binary components retain their own notices.
- [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE): Apache-2.0;
  [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt): BSD-3-Clause.
- Optional geometry/review uses original MIT/Apache-2.0 utilities from the
  sibling FoundationPose package, plus [trimesh](https://github.com/mikedh/trimesh/blob/main/LICENSE.md)
  (MIT). No FoundationPose neural model or original noncommercial research
  implementation is used by this tool.
- Optional training-data review uses [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE),
  MIT-CMU, for bounded image-header inspection, plus trimesh as listed above.
- [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt), BSD-3-Clause,
  is used only by the developer's independent pose-conversion reference check.

## Real examples

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
WACV 2017. [Publisher/license](https://cmp.felk.cvut.cz/t-less/): CC BY 4.0.
Selected original RGB/depth/mask/CAD/calibration bytes and pose annotations
retain CC BY 4.0. Derived bounding rectangles, metric conversion and overlays
are identified as modifications. In the earlier inference studies, pose annotations are evaluation-only.
The new training-data review explicitly uses official training-acquisition/PBR
labels for data preparation, never as inference input. Its synthetic training
and real-camera validation roles, converted labels and diagnostic graphics
are identified as modifications; no network training is performed.
No endorsement is implied.
