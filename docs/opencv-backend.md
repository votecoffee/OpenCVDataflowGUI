# OpenCV runtime and browser options

The project currently supports `opencv-python>=4.10,<5`. The local development
environment was verified as OpenCV **4.14.0** on 2026-11-04. Its desktop UI is
PySide6/Qt, with native OpenCV Python bindings and NumPy image buffers. It does not
embed a browser or execute WebAssembly.

OpenCV's [official releases](https://opencv.org/releases/) list **5.0.0**. Expanding
the supported major version requires a compatibility pass over native wrappers,
saved examples and Python packages. In particular, the
[OpenCV 5 changes](https://github.com/opencv/opencv/wiki/OpenCV-5) move G-API into
`opencv_contrib`; the current app has real G-API blocks. The migration needs to
verify the intended wheel's G-API bindings and image/geometry APIs, then run the
existing local tests and example jobs before changing dependencies. No OpenCV 5
compatibility is claimed by this change.

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
