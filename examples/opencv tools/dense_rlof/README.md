# Dense RLOF optical flow

Open [dense_rlof.json](dense_rlof.json) and choose **Run All**. The first frame
gives both flow blocks a reference. On the second frame, **Fixed support** and
**Cross support** estimate the same translation with different local windows.
The 120 × 96 color texture moves three pixels right and two pixels down.
Both blocks publish the center 72 × 48 rectangle, defined on the previous
image grid.

| Support | Mean endpoint error | Round-trip agreement within 0.10 px |
| --- | ---: | ---: |
| Fixed | 0.00140 px | 3,456 / 3,456 |
| Cross | 0.00086 px | 3,456 / 3,456 |

Select a block to inspect the flow preview, **Consistency mask**, or **Consistency
error**. Flow vectors use local pixels with positive x right and positive y
down. The status includes the previous and current filename tickets. The mask
and region restrict the published field and diagnostics after motion estimation.

The block converts supported intensity and color inputs to 8-bit BGR.
Grayscale is replicated across channels and alpha is discarded. It offers
geodesic, EPIC and RIC interpolation. A 1 × 1 grid with seed rejection turned
off uses the local estimates directly, bypassing interpolation and smoothing.
Geodesic can reject seeds using a sparse round trip. EPIC and RIC need their
full seed grid because the interpolation routines expect enough neighbors.
If RIC reports its native heap limit on an image, reduce **RIC neighbors**.
The global motion prior needs an image of at least 40 × 30 pixels.

The dense consistency outputs use a separate reverse estimate. Even a zero
displacement can agree with its reverse in an area without useful texture.
Compare against known motion when qualifying a graph. The numerical results
above were measured with the OpenCV 5.0.0 contrib wheel. [OpenCV optflow reference](https://docs.opencv.org/5.0/extra_modules/optflow.html)

The synthetic images and graph are original and covered by this repository's
MIT license. Run the following from the repository root with the application
installed to regenerate them.

    python "examples/opencv tools/dense_rlof/build_example.py" --output "examples/opencv tools/dense_rlof"
