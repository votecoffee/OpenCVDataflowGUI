# Third-party notices and model provenance

Original package code is MIT, copyright 2026 votecoffee. The crop, depth
normalization and pose-update conventions were adapted from NVIDIA's deployment
files at TAO revision `9a30f9692bf29fb728520e9dba1c79be2bf65e74`:

- `tao_triton/python/model/foundationpose_model.py`: MIT notice, copyright
  2022 NVIDIA CORPORATION. Preserve [the notice](licenses/NVIDIA_TAO_MIT.txt).
- `tao_triton/python/postprocessing/foundationpose_postprocessor.py`: explicit
  Apache-2.0 header, copyright 2023 NVIDIA CORPORATION. `preprocess.py` retains
  this attribution and follows [Apache-2.0](licenses/Apache-2.0.txt), even though
  the upstream repository's root license is MIT. Per-file terms matter.

These references were ported to bounded CPU NumPy/OpenCV processing. The pilot
does not vendor/import the original NVlabs research implementation, CUDA Warp,
NVDiffRast, PyTorch3D, Open3D or Triton. The CPU triangle renderer is development
code, not a claim of numerical parity with the upstream CUDA renderer.
[Audited deployment source](https://github.com/NVIDIA-TAO/tao-toolkit-triton-apps/tree/9a30f9692bf29fb728520e9dba1c79be2bf65e74).

## NVIDIA model assets

**FoundationPose**, Bowen Wen, Wei Yang, Jan Kautz and Stan Birchfield, CVPR 2024.
[Official project](https://nvlabs.github.io/FoundationPose/) and
[published NVIDIA models](https://huggingface.co/nvidia/foundationpose).

The two ONNX files are pinned to model revision
`18d8309afc9790cddc03a1d50bc69954dc058693` under the model card's
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
They are not embedded in this source repository or wheel. Explicit fetching
verifies size and actual SHA-256 before loading. This does not relicense the
separate original noncommercial FoundationPose research code. Model scores are
cohort-dependent ranking logits, not probabilities.

## Libraries and data

NumPy, ONNX Runtime, Trimesh and Pillow use their own permissive licenses;
OpenCV retains its distribution/license notices. Install them separately;
this package does not vendor their source or binaries. ONNX is used only for
optional development checks. GPU/native deployment dependencies remain a
separate future audit.

The real pilot uses **T-LESS**, T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas,
M. Lourakis and X. Zabulis, WACV 2017, under **CC BY 4.0**.
[Publisher](https://cmp.felk.cvut.cz/t-less/) and
[license](https://creativecommons.org/licenses/by/4.0/).
Retained reference recipes, derived clouds/images and example results keep that
data license and adjacent attribution. Ground-truth instance masks assist the
pilot; object poses are used only for evaluation. No endorsement is implied.
