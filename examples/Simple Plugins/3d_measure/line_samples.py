"""Generated first-visible line traces with independent noise and unrelated samples."""

import numpy as np
from opencv_simple_3d_measure import LineRegion, Measurement


def line_cloud(seed=1, *, missing=False, shortened=False, tilt=0, gap=4):
    rng = np.random.default_rng(seed)
    parameter = np.linspace(-10, 10, 600)
    first = np.c_[parameter, np.zeros(600), np.zeros(600)]
    second = np.c_[parameter, gap + parameter * np.tan(np.deg2rad(tilt)), np.zeros(600)]
    theta = np.deg2rad(30)
    third = np.c_[parameter * .7 * np.cos(theta), 8 + parameter * .7 * np.sin(theta),
                  np.full(600, 3)]
    if shortened:
        first = first[abs(parameter) < 7]
    parts = [first, third] if missing else [first, second, third]
    data = np.concatenate(parts)
    data += rng.normal(0, .015, data.shape)
    outliers = rng.uniform([-11, -.4, -.4], [11, .4, .4], (60, 3))
    return np.r_[data, outliers]


def line_configuration(scale=1):
    def region(name, lo, hi):
        return LineRegion(name, tuple(np.asarray(lo) * scale), tuple(np.asarray(hi) * scale))

    lines = (region("datum", (-12, -.5, -.5), (12, .5, .5)),
             region("target", (-12, 3.4, -.5), (12, 4.6, .5)),
             region("angled", (-8, 4.2, 2.5), (8, 11.8, 3.5)))
    measurements = (
        Measurement("gap", "line_separation", "datum", "target", lower=3.9*scale, upper=4.1*scale),
        Measurement("angle", "line_angle", "datum", "angled"),
        Measurement("probe", "point_line_distance", "datum", point=(0, 3*scale, 4*scale)),
        Measurement("span", "line_span", "datum"),
    )
    return lines, measurements
