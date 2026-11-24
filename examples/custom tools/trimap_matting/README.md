# Trimap alpha matting

Open [trimap_matting.json](trimap_matting.json) and choose **Run Once**.
The synthetic gasket has feathered edges and openings. The black trimap pixels
set alpha to 0, white pixels set it to 1, and gray pixels mark the region to solve.

The **Alpha** output is a float32 image on the composite's pixel grid. The
**Cutout** output attaches that alpha to the input colors as straight alpha.
The **Report** gives the number of unknown pixels, iterations, and solver residual.

On this fixture the mean absolute error over 3,908 unknown pixels is
approximately **0.00071** on the 0–1 alpha scale. The known pixels retain the
exact opacity specified by the trimap. The original analytic opacity is
stored separately in [truth.png](assets/truth.png) as full-range 16-bit grayscale.

The block uses the local color model from [Levin, Lischinski, and Weiss](
https://people.csail.mit.edu/alevin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)
with hard trimap constraints and a SciPy sparse solve. The OpenCV 5 wheel
used here lacks the Eigen-enabled information-flow matting module.

The bundled PNGs are original MIT-licensed assets.
Run `build_example.py --output <directory>` to regenerate the images and project.
