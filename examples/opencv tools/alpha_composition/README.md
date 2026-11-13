# Source-over alpha composition

Open `project.json` and run once. It needs the standard application installation.
Three branches compose an original 16-bit overlay using straight RGBA,
premultiplied RGBA, or a three-channel image with a separate alpha map. Each is
checked against an independent scalar reference evaluated with exact rational
arithmetic. The allowed maximum error is one uint16 sample unit.

The result retains background transparency. Checkerboard previews show the
transparent areas. The premultiplied PNG is an intentional processing fixture;
its block explicitly selects premultiplied input. Standard output is straight
BGRA, suitable for ordinary PNG interchange.

From the repository root:

```powershell
opencv-dataflow-run "examples/opencv tools/alpha_composition/project.json"
opencv-dataflow-run "examples/opencv tools/alpha_composition/project.json" --profile lean
```

Regenerate the original layers and rational reference with:

```powershell
python "examples/opencv tools/alpha_composition/generate.py"
```

## Composition and alpha representation

**Source-over alpha composition** applies the
[W3C source-over equations](https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators)
on the CPU. For straight input colors, output alpha is
`Ao = Af + Ab * (1 - Af)` and associated color is
`Co = Cf * Af + Cb * Ab * (1 - Af)`. Straight output divides associated color
by positive output alpha. Fully transparent selected output pixels have zero
color, including pixels whose alpha rounds to zero during integer storage.

Straight colors are independent of alpha. Premultiplied colors already include
their embedded alpha. The Auto input settings read `alpha_representation`
payload metadata and otherwise assume straight input. Output records this
association so another compositor can consume it automatically without applying
embedded alpha twice. Premultiplied inputs and output require RGBA/BGRA, and
associated input colors must be zero at zero alpha.

The optional alpha map multiplies embedded foreground alpha. Global opacity
multiplies both that effective alpha and the associated foreground color. A
three-channel foreground starts opaque, which makes the separate-map branch
equivalent to the embedded-alpha branches here.

Integer premultiplication introduces quantization. Premultiplied PNG transport
also needs an explicit association setting when reopened; image-file loading
defaults to straight alpha. Choose straight output for normal PNG interchange.

## Pixel types, colors and selection

Foreground and background must already occupy equal-size registered grids.
Gray, RGB/BGR and RGBA/BGRA inputs can use different supported depths and channel
orders. Integer color and alpha values use their unsigned full scale; floating
alpha and external maps use [0,1]. Stored-space floating colors can retain finite
HDR values. HSV/Lab and physical depth planes require an explicit conversion
before this operation.

Output preserves the background format/depth by default. Explicit choices allow
gray, RGB/BGR, RGBA/BGRA and uint8/uint16/float32/float64. Premultiplied output
retains an alpha channel. Dropping alpha into a three-channel output keeps
straight color; the separate alpha output still reports the computed opacity.
Integer output rounds and saturates; `clipped_samples` reports range clipping.

Stored mode blends the component values directly. Linear sRGB mode decodes unit
sRGB colors before blending and encodes the result afterward. Half-white over
black is 128 in stored uint8 space and 188 with linear sRGB blending. HDR values
outside [0,1] require stored mode or prior color processing.

Masks and circle/rectangle/polygon regions select output centers on the
background grid. Outside the selection, pixels represent the background in the
chosen storage format and association. When those match the original background,
its unselected samples are copied exactly. Opaque/transparent foreground
endpoints also preserve exact compatible background/foreground samples.

## Outputs and context

| Output | Meaning |
| --- | --- |
| `image` | Composited image with encoding, depth and alpha-association metadata. |
| `alpha` | Float64 computed output alpha in [0,1], before storage quantization. |
| `info` | Resolved input/output associations, formats, depths, color space, selected/active counts, clipping and source identities. |
| `foreground_alpha` | Embedded alpha times external map and opacity, zero outside the selection. |
| `preview` | Optional checkerboard view of the actual stored image, with a drawable label. |

The first three connectors are exposed by default. Graphics disabled skips the
preview and label; numerical outputs remain available. Images are bounded to
2.1 MP and 16,384 pixels per axis. Invalid alpha, layouts, associations, shapes
and nonfinite/unrepresentable results produce contained errors.

Same-frame alignment checks known coordinate identity. Pixel-grid alignment
explicitly pairs grids by index, as needed for these independent fixture files.
Known tickets must match by default. The background policy explicitly permits
cross-source composition and retains background ticket/frame/pyramid/companion
payloads, with both source identities in the report. An unticketed background can
adopt the foreground ticket. This example selects background ownership because
its static files are separate input sources.

## Attribution

All images, the generator and the project are original repository examples under
the repository's MIT license. No customer or downloaded images are used. The
algorithm reference is the W3C compositing specification linked above.
