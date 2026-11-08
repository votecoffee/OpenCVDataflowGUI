# SimplePathWidth · Industrial weld bead

This [project](project.json) measures three real weld images. Follow the
[package setup](../README.md#setup) with SimplePathWidth 0.1.0.dev2 or later.
Run three times to see the setup image, a held-out image with clear edges,
and a held-out image with difficult edges.

The recipe uses a fixed path through the upper weld crop, with **Nominal
width** set to 130 px and **Edge search radius** set to 25 px. These
settings restrict the search to bands 40–90 px from the path on each side.
Open **Train path width…** to see the saved reference and change the bands.
Inspect **Result / overlay** or the result filmstrip for paired edges, and
**Result / trend** for width and offset. The demonstration limits are
100–160 px width and 90% station coverage. They are recipe settings for
replay. The dataset does not supply a quality verdict for these frames.

| Frame | Result | Measured stations | Mean width error versus mask |
| --- | --- | ---: | ---: |
| `20230930_21h03m42s_B11_1.png` | Pass | 31/31 | 3.3 px |
| `20231113_19h46m30s_B13_3.png` | Pass | 31/31 | 5.4 px |
| `20231114_11h25m52s_B04_3.png` | Fail | 18/31 | 17.2 px |

The original masks are supplied for comparison. The graph takes only the
inspection images as input. [Source and license](SOURCES.md)

The [fixture pose replay](pose_replay/README.md) rotates and translates the
second weld, then uses the supplied pose to align the inspection path. It
also demonstrates the **Edge continuity** setting.
