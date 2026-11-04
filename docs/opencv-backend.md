# OpenCV runtime and browser options

The project requires `opencv-contrib-python>=5.0.0.93,<6`. The local development
environment uses OpenCV **5.0.0**, Python distribution **5.0.0.93**. Its desktop UI is
PySide6/Qt, with native OpenCV Python bindings and NumPy image buffers. It does not
embed a browser or execute WebAssembly.

## OpenCV 5 migration

The contrib wheel includes extra modules and retains the G-API functionality used
by existing grayscale/blur/Canny nodes. Their block IDs, parameters and saved
connections remain unchanged. GComputation now receives native argument lists
instead of the removed Python `GIn/GOut` helpers; no eager fallback is used.
Line-segment detection normalizes scalar-per-line output vectors before exposing
width, precision and NFA measurements.

The migration was checked against the complete automated suite, offscreen Qt
editor workflows and saved example graphs. Shape/ORB models trained and saved on
4.14.0 were also loaded and matched on 5.0.0 without retraining. This is synthetic
compatibility evidence; representative inspection images should determine actual
acceptance tolerances. OpenCV's [migration guide](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)
documents array-layout, interpolation and rendering changes, so numerical results
are not promised to be bit-identical across major versions.

Install only one OpenCV distribution in an environment. The base, contrib and
headless packages all provide `cv2`; contrib replaces the base wheel. See the
[README upgrade commands](../README.md#install). The dependency exposes additional
native APIs to plugins; new contrib algorithms still need explicit block wrappers
to appear in the toolbox.

## Browser option

WebAssembly is worth considering for a future browser-based editor, viewer or
portable preview runner. It is not new to version 5:
[OpenCV.js builds](https://docs.opencv.org/4.x/d4/da1/tutorial_js_setup.html) already
target WebAssembly in OpenCV 4. A browser backend could process selected images
locally without a Python service, but would need an explicit supported-block
subset, graph/model interoperability and measurements of latency and image-copy
costs. Python plugins, device acquisition and filesystem/database writers would
need separate handling.

For the current native desktop runtime, there is no demonstrated benefit from
adding a WebAssembly execution layer. Keep this as a future backend option and
benchmark a concrete browser workflow before introducing it.
