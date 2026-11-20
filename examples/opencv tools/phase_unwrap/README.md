# Unwrap measured fringe phase

Open [phase_unwrap.json](phase_unwrap.json) and run once. The wrapped phase
was demodulated from three synthetic 16-bit camera exposures, spaced by
120 degrees. The lit mask keeps two independent regions on either side of
the shadow strip. A small dark patch and one missing pixel stay masked.

The **Relative phase per region** branch retains each region's own cycle
zero. **Absolute references per region** uses one phase reference on the
left and two on the right. The integer offsets are −3 and +4 cycles.
All 8,127 supported pixels are recovered. Against the independently known
synthetic surface, RMS error is 0.0000106 rad and the largest error is
0.0000284 rad.

The phase and native inverse-reliability outputs preserve the phase
image's pixel grid. Smaller inverse reliability means smoother local
phase. Masked pixels are NaN, including in the diagnostic image output.
In the previews they appear black. **Component Labels** identifies the
four-connected groups and **Valid Mask** identifies the phase emitted
after the references have been applied.

Supply float radians in `[-π, π]` and a matching uint8 mask with nonzero
values for the illuminated area. An optional region limits the same
selection. Supply references as a list of records with `pixel` set
to `[x, y]` and `phase_rad` set to the known absolute phase at that
pixel. A wired list overrides the configured references. Relative mode
uses the first selected pixel's wrapped cycle if there is no reference.
Required mode withholds that component. References on the same component
must agree on the integer offset and be within the anchor tolerance.

The algorithm assumes the phase varies by less than a half cycle between
sampled neighbors where the surface is continuous. Keep shadows and
discontinuities outside the support mask. The histogram controls are
passed to OpenCV after validation. [OpenCV phase unwrapping API](https://docs.opencv.org/5.0/extra_modules/classcv_1_1phase__unwrapping_1_1HistogramPhaseUnwrapping.html)

The original exposures and analytic reference are synthetic fixtures by the
OpenCVDataflowGUI contributors (MIT). Rebuild with
`python build_example.py --output <new-directory>`.
