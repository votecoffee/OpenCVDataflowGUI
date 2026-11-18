# ARMBench photo attribution

Photographs by **Amazon.com, Inc.**, from the **ARMBench Sparrow Object
Segmentation** tiny sample archive, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The benchmark is described by Chaitanya Mitash, Fan Wang, Shiyang Lu,
Vikedo Terhuja, Tyler Garaas, Felipe Polido, and Manikantan Nambi in
*ARMBench: An Object-centric Benchmark Dataset for Robotic Manipulation*,
ICRA 2023. [Publisher and license](https://www.armbench.com/) ·
[Paper](https://arxiv.org/abs/2303.16382)

| Archive member under `armbench-segmentation-tiny/` | Use in this project |
| --- | --- |
| `P2S0835EOA.jpg` | Training, mixed inventory |
| `9TARTKRF1X.jpg` | Training, stacked packages |
| `OAR1KXNIME.jpg` | Validation, loaded tote |
| `EZ5895NGB8.jpg` | Validation, empty tote |

Amazon's [pinned tiny archive](https://huggingface.co/datasets/amazon/armbench-segmentation/resolve/4444a1cc2c592afc3eea2b7da63aff11328e9aea/archive/armbench-segmentation-tiny.tar.gz)
has SHA-256 `30f2539a2c8ae32ae25000ee9025657e2c49529cf9e513196597f9a94e7ca0d7`.
[Archive distribution license](https://huggingface.co/datasets/amazon/armbench-segmentation/blob/4444a1cc2c592afc3eea2b7da63aff11328e9aea/License-CC-BY.txt)

The example resizes the original 2448 × 2048 photos to 612 × 512 using
area interpolation and re-encodes them as JPEG at quality 93. The grayscale
PNG files use the same blue-dominance transform as the saved graph.
The binary labels come from polygons traced on the photographs for this
example. The tiny archive contains photographs only.

After downloading the pinned archive, run this command from the repository root.

```powershell
python "examples/Simple Plugins/pixel_segmentation/armbench_tote/build_example.py" path/to/armbench-segmentation-tiny.tar.gz path/to/new-example
```

The `result_views/` PNGs are derived from the same loaded/empty validation
photos after the saved blue-dominance transform. Added colors display the
unchanged foreground mask, inspected region and fixed-scale forest votes.
