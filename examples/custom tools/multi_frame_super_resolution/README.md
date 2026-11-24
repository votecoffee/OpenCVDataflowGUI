# Classical multi-frame super-resolution

Open [estimated shifts](super_resolution_estimate.json) or
[known shifts](super_resolution_provided.json) and choose **Run all**. The
five 96 × 72 observations contain small translations of the same textured
plate. The batch reconstructs a 192 × 144 view around frame 03. Both graphs
compare the result with bicubic interpolation of that frame and with the
original high-resolution target.

The saved estimated-shift run accepts all five observations and finds all
four 2× phase bins. Its measured interior error is **2.99** intensity levels
RMS, compared with **20.74** for bicubic. The nominal shifts are half a source
pixel or zero. Estimated shifts agree within 0.01 source pixels. The
known-shift graph reports **3.09** intensity levels RMS on the same interior.
The test scene includes one-pixel dark marks on the moving plate.

The block uses CPU phase correlation and ECC for global translation.
For known alignment, supply one `[dx, dy]` pair per batch frame, measured
from the reference into that frame, with `[0, 0]` for the anchor. Provide the
raw observations so that their differing subpixel samples remain available.
The reconstruction minimizes a robust observation error through a translated
box-aperture model. A neighboring-pixel penalty controls noise and artifacts.

An odd window of 3–15 frames is selected around the reference. The block
reports rejected alignments, subpixel phase coverage and the number of complete
observation footprints per output sample. Insufficient aligned observations
or phase diversity withhold the reconstruction. Interpolation fills pixels
outside the selected, sufficiently covered area. The image inherits the
reference ticket and a nominal half-pixel mapping to its source raster.

For continuous video, set **Batch serial entries** to stride 1 on a single
serial input. It emits a rolling window for every new frame after warm-up.
A centered estimate arrives after the trailing half of its window has been
acquired. Reset flow state before changing clips or rewinding a rolling
window. In the known-shift workflow each window needs offsets relative to
its selected anchor.

The example evaluates a synthetic translation and box-sampling model.
Rotation, independently moving objects, blur that differs from the model,
and poor alignment can reduce image fidelity. The supported input depths are
uint8, uint16 and normalized float32. Straight alpha uses reference
interpolation. Output is bounded to 2,097,152 pixels and the iteration budget
depends on scale, channels, window and iteration count.

This project uses the installed OpenCV/SciPy CPU backend. The OpenCV 5.0
contrib Python wheel exposes its learned DNN super-resolution module. The
native BTV-L1 interface is documented separately for C++.
[OpenCV super-resolution reference](https://docs.opencv.org/5.0/extra_modules/superres.html)
[Classical observation and robust reconstruction reference](https://people.duke.edu/~sf59/srfinal.pdf)

The assets are original MIT fixtures. Rebuild them with
`python build_example.py --output <directory>`.
