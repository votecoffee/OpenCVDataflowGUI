# Reference-guided hard shadows

**Detect and compensate reference shadows** (`cv.reference_shadows`) compares a
current image to an aligned, unshadowed nominal reference. It models each detected
shadow as one scalar attenuation of all intensity channels. This is a useful
inspection hypothesis, not a general proof that a dark region is a shadow.

The block performs no registration. Match image size, sample depth and RGB/BGR
order; reference alpha may be omitted. **Pixel grid** permits reference captures
with a different ticket/root. **Same frame** also checks known coordinate grids.
Output tickets, timing, frame/pyramid and companion payloads belong to the current
image. Unsigned 8/16-bit and float32/64 intensity images retain sample depth,
encoding and straight alpha. Premultiplied images, physical depth planes,
negative/nonfinite samples and unsupported color spaces are rejected.

For each selected pixel, detection uses the current/reference arithmetic-mean
intensity ratio. Color images also compare each channel's fraction of total
intensity, with an absolute maximum channel-fraction error. Alpha is omitted.
**Min/Max ratio** bound attenuation, while **Max chroma error** rejects color
changes. **Minimum area** removes small connected candidates; choose it above
the size of defects you need to preserve as isolated marks. **Connectivity** is
4 or 8. No smoothing or feathering crosses the measured shadow boundary.

**Fill holes area** optionally includes small enclosed noncandidate regions in
the shadow mask. It is zero by default. Filling can correct the illumination of
an enclosed colored defect while retaining its samples, but it assumes the hole
shares its surrounding shadow. Holes touching unselected/invalid reference
pixels are never filled. Complementary connectivity prevents diagonal leaks.

For each retained region, **Fit erosion** removes boundary pixels from estimation
only. Fit support includes original candidates, excludes filled holes and is
intersected with the optional uint8 **Fit mask**. This mask must be on the current
image grid; it does not restrict where the fitted gain is applied. The gain is
`1 / median(ratio)`. **Minimum fit pixels**, **Max gain** and **Max ratio spread**
qualify the estimate. Spread is `(q90 - q10) / median`. The same scalar gain
multiplies all original pixels in the region, retaining local texture and defect
contrast. Reference pixels are never copied into the result.

| Output | Meaning |
| --- | --- |
| Image | Corrected image, or no event if the model is invalid. Pixels outside the detected regions and alpha are preserved exactly. |
| Shadow mask | Detected retained regions, including configured hole filling; also emitted for rejected models. |
| Invalid mask | Selected unusable reference/ratio pixels and rejected shadow components. |
| Clipped mask | Pixels where a qualified gain exceeds output sample range, including rejected clipping attempts. |
| Gain map | Float64 scalar gain, one outside corrected regions. Withheld when Image is withheld. |
| Valid | Whether the selected correction is usable; this is not a final inspection judgment. |
| Measurements | Counts and per-component bounds, support, median ratio, gain, spread and status. |

**Min reference** is a fraction of full scale for unsigned images and raw sample
units for floats. Zero/dim references and nonfinite or unrepresentable ratios are invalid evidence.
All retained components must qualify, and all selected references must be usable.
An invalid or empty selection withholds Image/Gain and emits masks, structured
failure reasons and measurements. A valid image with no candidates is returned
unchanged; it does not mean the part passed its inspection.

**Clip policy = Reject** withholds correction if any modeled sample exceeds the
output depth. **Clip** explicitly permits saturation and reports its coverage.
Floating images retain raw units and are not clipped to 0..1. Integer conversion
rounds to nearest. Optional drawing objects show fitted/rejected region bounds;
Graphics Off skips their construction. Diagnostic arrays remain ordinary outputs
in Lean because they are part of the block's result contract.

Work is bounded to 4,194,304 image pixels, a configurable maximum of 1..4096
retained components (default 256), and 33,554,432 cumulative component bounding-box
pixels. Component reports and failure codes are bounded independently. The block
has no mutable sequence state and can run on parallel branches.

Large neutral dark material changes, occlusion, misregistration, differing camera
response, clipped captures and colored shadows can violate the model or be
misclassified. The spread limit can also allow residual illumination variation.
Use trusted fit support and review the masks on representative captures. This
implementation does not estimate a penumbra or spatially varying illumination.

The [original replay](<../examples/custom tools/reference_shadows/README.md>)
qualifies exact neutral hard-edge correction and defect retention, with an
independent invalid-fit case. OpenCV references:
[connected components](https://docs.opencv.org/5.0/main_modules/imgproc_shape.html)
and [erosion](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html).
