# Fast bilateral solver

Open [sparse_signal.json](sparse_signal.json) and choose **Run Once**. The
guide separates three vertical bands. The signal supplies six measured patches
in the first two bands, and the confidence image marks their 150 pixels.
Select **Reconstruct measured bands** to inspect the output and its support
mask.

| Guide band | Result | Supported pixels |
| --- | --- | ---: |
| Left | Recovered intensity 53 at every pixel | 4,608 |
| Middle | Recovered intensity 187 at every pixel | 4,608 |
| Right | Kept the input value 83 | 0 |

The block preserves the signal's numeric depth and alpha. The confidence port
takes a monochrome uint8 or normalized float32 image. A mask or region limits
the pixels the block replaces. The solve can still use confident samples
elsewhere in the aligned input. An unsupported guide component keeps its
original pixels and appears black in **Supported**.

The default **Auto** backend uses the SciPy solver with the project's standard
OpenCV 5.0 wheel, which was built without Eigen. An Eigen-enabled build can
run the **OpenCV** backend on guides within its coordinate range and with
confidence in every connected guide component. The report shows which backend
ran. The SciPy path also reports convergence and per-channel residuals. A
nonconverged solution keeps its current estimate and records a failed block
judgement.

The example images and graph are original MIT assets. The SciPy backend
implements the sparse bilateral-space optimization described by Barron and
Poole. [The Fast Bilateral Solver](https://research.google/pubs/the-fast-bilateral-solver/)

The native build requirement is defined in the OpenCV source.
[OpenCV contrib solver source](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/ximgproc/src/fbs_filter.cpp)
