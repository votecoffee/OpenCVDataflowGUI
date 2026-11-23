# Dual TV-L1 optical flow

Open [dual_tvl1_flow.json](dual_tvl1_flow.json) and choose **Run All**.
The first frame primes both sequence blocks. The second gives each detector
the same moving textured patch against a stationary background. Open their
results to compare the flow preview, consistency mask, error and status.
Reopen the project or reset its sequence state before replaying the pair.

The patch moved five pixels right and three pixels up. Both branches estimate
flow over the whole image and publish vectors in the marked 33 × 30 interior
region. On the fixture, the mean endpoint error was **0.0086 pixels** for
Dual TV-L1 and **0.0046 pixels** for Farneback. All 990 selected points passed
the 0.5-pixel forward–backward agreement check with each solver.

**Flow** stores float32 `[dx, dy]` vectors on the previous image grid.
**Consistency mask** marks selected vectors that agree with the reverse
estimate. **Consistency error** gives the round-trip error in pixels and
uses NaN for unevaluated locations. This measures agreement between the two
directions. The known synthetic translation is what establishes accuracy
for this example.

Dual TV-L1 exposes the solver weights, pyramid scale, warps, inner and outer
iteration limits, stopping tolerance, illumination term, median kernel, and
optional seeding from the previous unmasked field. The reverse consistency
solve starts independently. Image conversion preserves 8-bit grayscale and
uses normalized float32 for higher-depth input. The block bounds the
estimated native workspace separately from the configured sequence memory.

Both PNGs are original MIT-licensed synthetic fixtures. Rebuild with
`build_example.py` in the application environment.
[OpenCV Dual TV-L1 reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1optflow_1_1DualTVL1OpticalFlow.html)
