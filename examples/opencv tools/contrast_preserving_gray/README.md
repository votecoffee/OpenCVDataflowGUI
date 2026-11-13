# Preserve color separation when converting to gray

Open [contrast_preserving_gray.json](contrast_preserving_gray.json), select
**Debug**, and run once. Two color patches both become **92 DN** under ordinary
8-bit BT.601 conversion. The original seeded color context lets native
`cv2.decolor` separate their means by **77 DN** on the qualified build.
The two contrast measurement blocks report zero before and at least 50 DN after.

Inspect these outputs from **Contrast-preserving grayscale**:

| Output | Meaning |
| --- | --- |
| `image` | Native contrast-preserving 8-bit grayscale inside the selection. |
| `color_boost` | Native enhanced color, retaining original channel order and alpha. |
| `ordinary_gray` | Ordinary BGR/RGB-to-gray baseline across the whole image. |
| `selected_mask` | Intersection of optional input mask and region, or the entire grid. |
| `change` | Absolute difference between output grayscale and the ordinary baseline. |
| `report` | Fitting context, selection count, native range, status and no-op reason. |

## Mapping and selection

The native function exposes no tunable algorithm parameters. It learns its
mapping from the **whole input image**; mask and region restrict the writes, not
the fitting samples. Outside the selection, `image` retains ordinary grayscale
and `color_boost` retains the original color samples. Holes follow the same rule.
The mapping is content-dependent: output brightness and polarity can change
with surrounding colors. It is not a calibrated luminance measurement or a
fixed mapping that can be trained here and reused on later frames.

Use uint8 BGR/RGB or straight-alpha BGRA/RGBA, at least 8×8 and up to
2.1 million pixels. Grayscale outputs drop alpha; `color_boost` preserves it
byte for byte. Stored color samples, including those under transparency,
participate in fitting; alpha is not a fitting weight. Premultiplied input must
first be unassociated. UInt16/float require explicit conversion. Coordinates,
ticket, acquisition data and other payloads remain attached.

## Inspect a known native failure

Open [degenerate_mapping.json](degenerate_mapping.json). It uses the same patch
colors without the varied fitting context. On the supported OpenCV 5 build,
native decolorization collapses this nonuniform image to flat black.
The block reports **`degenerate_native_mapping`** as a failed result and withholds
`image`, `color_boost` and `change`. The downstream converted-image measurement
does not execute. The ordinary baseline, selection and report remain available.
The headless runner intentionally exits 1 for this project.

Uniform-color inputs, already-neutral color images and empty selections bypass the solver, returning
ordinary grayscale and original color with an explicit no-op reason. The
positive fixture establishes color separation for these samples; it does not
claim that every equal-luminance pair will become separable.

## Source and reproduction

Both fixtures and graphs are original project work under MIT. No external or
customer images are included. [truth.json](truth.json) stores independently
defined colors, patch locations and the separation limit.

```powershell
python "examples/opencv tools/contrast_preserving_gray/build_example.py"
opencv-dataflow-run "examples/opencv tools/contrast_preserving_gray/contrast_preserving_gray.json" --profile debug
```

[Native API](https://docs.opencv.org/4.x/d4/d32/group__photo__decolor.html) ·
[OpenCV examples](../README.md)
