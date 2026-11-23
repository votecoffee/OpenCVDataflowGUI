# Anisotropic diffusion

Open [anisotropic_diffusion.json](anisotropic_diffusion.json) and choose
**Run Once**. The noisy 8-bit input has a known 121-level vertical step.
Compare the one-pass and ten-pass outputs. The larger-K branch shows how
stronger cross-edge diffusion softens that boundary.

| Image | RMSE against the clean target | Measured step |
| --- | ---: | ---: |
| Noisy input | 11.43 | 120.60 |
| One pass, K = 0.12 | 7.13 | 120.90 |
| Ten passes, K = 0.12 | 1.75 | 120.90 |
| Ten passes, K = 0.30 | 4.90 | 48.03 |

The measured step averages the difference between columns 80 and 81 across
all rows. The region branch applies the ten-pass result inside the rectangle
and retains the original pixels outside it.

**Time step** is bounded at 1/8 so the eight-neighbor update stays stable.
**Edge sensitivity** is OpenCV's K in normalized intensity units. Smaller
values limit diffusion across stronger intensity differences. Zero
**Iterations** copies the source. The output preserves the image encoding,
sample depth and alpha.

The block calls OpenCV once per native 8-bit pass to rebuild the replicated
border. Full-range byte images use an explicit update so the largest
three-channel difference is handled safely. Unsigned 16-bit and normalized
float images use the same update at higher precision. [OpenCV 5 diffusion source](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/anisodiff.cpp)

The project, builder and synthetic PNGs are original MIT assets. Regenerate
them from the repository root with the example builder.

    python "examples/opencv tools/anisotropic_diffusion/build_example.py" --output "examples/opencv tools/anisotropic_diffusion"
