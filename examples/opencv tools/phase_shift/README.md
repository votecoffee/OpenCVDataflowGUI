# Absolute projector coordinates from phase-shift captures

Open [projector_x.json](projector_x.json) or [projector_y.json](projector_y.json)
and click **Run All**. The TIFF source supplies six simulated captures in
chronological order. The batch waits for one complete phase sequence before
the decoder emits its projector map. Select the decoder to see the map,
wrapped and absolute phase, normalized modulation, cycle error and validity.

Each project recovers **7,167 of 7,776** camera samples. Maximum absolute
coordinate error is below **0.001 projector pixel** on these simulated
captures. The decoder rejects 510 low-signal samples and 99 inconsistent
coarse/fine samples. Both sides of the shadow stripe retain their absolute
projector coordinates.

## Use a capture sequence

Set the projector size, encoded axis and fine periods on **Generate
phase-shift patterns**. Project and capture the six images in order.
The first triplet is a single coarse fringe cycle, and the second is
the fine fringe. Both use 0°, 120° and 240° phase offsets. Use the same
specification when decoding. X coordinates use vertical stripes and Y
coordinates use horizontal stripes. Run separate sequences to get both
axes.

Keep the scene and camera still across the six exposures. Use aligned
monochrome uint8, uint16 or normalized floating intensity. Captures can
have a different size from the projector. Tagged sequences are checked
for matching order and specification. Untagged sequences use the received
order. A mask or region limits processing to selected camera pixels.

The coarse phase chooses the integer cycle of the fine phase at each
pixel. Adjust the minimum exposure, minimum modulation, and maximum
cycle disagreement for the camera signal. Optional saturation rejection
masks pixels that reach either intensity rail. Invalid coordinates are
NaN, and the uint8 validity mask uses the camera grid. Coordinates are
projector pixels on the selected axis. For metric 3D they need calibrated
camera/projector geometry.

OpenCV 5.0.0's generator raises an exception below three requested
cycles. The block evaluates the same sine directly for those cycles.
Native fine-fringe generation and ideal PSP phase/unwrap behavior are
covered by tests. The deployed decoder uses temporal coarse/fine phase
for the absolute cycle. [OpenCV structured light](https://docs.opencv.org/5.0/extra_modules/structured_light.html).

The [fixture builder](build_example.py) recreates the captures and projects.
The label PNGs use 0 for valid, 1 for low signal and 2 for phase disagreement.
Original synthetic fixtures by OpenCVDataflowGUI contributors under
[MIT](../../../LICENSE).
