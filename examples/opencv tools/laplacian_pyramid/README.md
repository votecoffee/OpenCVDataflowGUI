# Laplacian image pyramids

Open `project.json` and **Run once**. The original **193 × 129, 16-bit color**
fixture splits into three signed detail bands plus a coarse base. The upper
reconstruction branch restores the original pixels exactly; the verification
block must report **maximum absolute error 0**. The lower branch removes the
finest detail band to show how multiscale editing changes the image.

The image, generator and project are original synthetic content under the
repository's MIT license. Regenerate from the repository root with
`python "examples/opencv tools/laplacian_pyramid/generate.py"`.

**Laplacian image pyramid set** (`cv.laplacian_pyramid`) outputs the existing
`pyramid` connector type. Levels run largest first. Each detail is
`G[level] - pyrUp(G[level+1])`; the final level is the coarse Gaussian base.
Use **Select pyramid level** to inspect a band, or connect the set to an image
input and select its level. The signed diagnostic preview maps zero to neutral
gray. Raw coefficient images retain negative values and original sample units.

The default floating backend uses float64 coefficients and supports gray,
BGR/RGB and BGRA/RGBA uint8, uint16, int16, float32 and float64 input. Float32
working depth is an explicit precision/memory tradeoff. Both the level selector
and source payload identify the local pixel grid and scale. Reconstruction
restores the original source frame and any incoming pyramid context.
Floating inputs reconstruct within the working precision; arbitrary dynamic
ranges do not have a bitwise-identity guarantee.

**Reconstruct Laplacian pyramid** (`cv.reconstruct_laplacian_pyramid`) uses the
recorded dimensions at every expansion, including odd widths/heights. `source`
output depth restores the original dtype without scaling intensities. Integer
outputs round and saturate, with a clipped-sample count in the report. Floating
outputs retain raw units: a 16-bit image reconstructed as float64 remains on
its 0–65535 scale. Optional detail gains are largest first; zero removes a band.
Default gains are all one. The coarse base is retained.

The optional **native_detail** backend wraps OpenCV's `createLaplacePyr` and
`restoreImageFromLaplacePyr`. It requires uint8 three-channel input, uses signed
int16 bands and fixed reflect-101 borders, and supports unit-gain reconstruction.
Use the floating backend for other image depths, alpha, or detail-gain edits.
Native reconstruction works on private copies so it cannot alter another branch.

Gaussian sets, incomplete levels, incompatible geometry and invalid gain lists
are rejected. Pyramids are limited to nine levels and 8,388,608 base pixels.
The CPU backends are covered here; GPU detail routines remain outside this block.

API background: [OpenCV image pyramids](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html).
