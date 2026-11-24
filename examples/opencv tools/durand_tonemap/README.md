# Durand HDR tone mapping

Open [durand_tonemap.json](durand_tonemap.json) and choose **Run Once**. Compare
**Global linear mapping** and **Durand local mapping**. The left band has
fine texture in shadow, the right band has texture in bright light, and the
filmstrip shows the local result.

| Band | Linear | Portable Durand |
| --- | ---: | ---: |
| Shadow | 0.00116 | 0.01060 |
| Midtone | 0.01431 | 0.04173 |
| Highlight | 0.17295 | 0.16288 |

Values are the difference between adjacent bright and dark texture stripes,
measured away from band boundaries in the float32 display luminance. The
portable backend increases the visible shadow difference about ninefold on
this fixture. Both branches run at gamma 1.

The block accepts nonnegative float32 radiance in gray, BGR or RGB. It
preserves channel order, image geometry and ticket. Select float32 to see the
display levels in [0, 1] or uint8 for a display-ready image. Keep the source
branch for radiometric measurements.

**Portable** is the default. It limits the contrast of the bilateral base in
log2 stops and adds the local luminance detail back. The range sigma also
uses stops. [Durand and Dorsey's bilateral tone mapping](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/)

**OpenCV** uses the native implementation. Both contrast and range sigma
use natural-log units, and spatial sigma is in pixels for both backends.
The factory requires an enabled contrib build. On a separate OpenCV 5.0.0
build with `OPENCV_ENABLE_NONFREE`, the example fixture's shadow-stripe
difference was 0.02661 at the native defaults, with exact agreement against
the direct native call after clipping. [OpenCV xphoto reference](https://docs.opencv.org/5.0/extra_modules/xphoto.html)

The TIFF holds the original float32 samples and is an original MIT-licensed
fixture. Run `python build_example.py --output <directory>` to recreate it
and the project.
