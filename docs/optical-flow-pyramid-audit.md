# Precomputed optical-flow pyramid binding audit

S1-155 remains **Planned**. Building a pyramid is available, but the supported
Python tracker cannot yet consume that result as a reusable tracking cache.
The existing generic image-pyramid blocks remain useful for multiscale image
processing; they do not implement this native tracking optimization.

On the supported OpenCV 5.0.0 contrib wheel, `buildOpticalFlowPyramid` returns a
tuple of NumPy images, optionally interleaved with int16 two-channel derivatives.
For a 161×129 uint8 image, 21×21 window and requested maximum level 3, the builder
returns maximum level 2, with sizes 161×129, 81×65 and 41×33. Returned image
level zero does not share NumPy storage with the input in this probe, even with
`tryReuseInputImage=True`.

Both `calcOpticalFlowPyrLK` and `SparsePyrLKOpticalFlow.create().calc` reject all
12 tested combinations: tuple, list or tuple of UMat values, each with and
without derivatives. The reported argument errors require an image/UMat rather
than this Python pyramid representation. The C++ API's ability to accept
prebuilt pyramids does not establish support through this wheel's Python API.

Reproduce without a camera or GUI:

```powershell
python scripts/audit_optical_flow_pyramids.py --report NEW_REPORT_PATH.json
```

The report includes version, controls, returned levels/types/ownership, unchanged
input and individual consumer outcomes. It preserves an existing report path.
The test checks representation and records any future identity-tracking success;
it does not demand that a newer binding continue rejecting pyramids. Successful
binding qualification alone would still require a graph implementation, cache
invalidation/ownership limits and direct-versus-cached parity and performance
evidence before completing S1-155. A qualified native bridge or tracker-owned
cache is a possible future implementation.

Upstream: [pyramid builder and sparse tracking API](https://docs.opencv.org/5.0/main_modules/video_track.html).
