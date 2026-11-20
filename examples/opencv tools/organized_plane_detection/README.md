# Find two planes on a depth grid

Open [organized_plane_detection.json](organized_plane_detection.json) and
run it. The synthetic scene contains a frontal surface at 0.90 m and a
tilted surface with camera-frame equation
`0.29 X − 0.12 Y − Z + 1.12 = 0` before normalization. A small curved
part sits in front of both surfaces. Two rectangular patches have
missing depth.

The block uses OpenCV organized normals and `findPlanes`, then refits
the native members in float64 and checks their geometric residuals. Its
plane records contain normalized camera-frame equations, support counts,
and residuals in meters. The 5 mm threshold is applied to the final
orthogonal distance. Native sensor-error coefficients affect the initial
search. [OpenCV point-cloud reference](https://docs.opencv.org/5.0/main_modules/ptcloud.html#findplanes)

Inspect **Plane labels** for the color preview. The raw uint16 **Labels**
output uses the following conventions.

| Label | Meaning |
| --- | --- |
| 0 | Valid depth outside the verified planes, including the curved part |
| 1 through 254 | Verified plane ID, ordered by decreasing support |
| 65535 | Invalid depth or pixels excluded by the selection |

**Unmatched depth** keeps the measured Z at label 0 and uses NaN
elsewhere. The separate **Valid mask** and **Unmatched mask** provide
binary selections on the same grid. A plane on another object can also
qualify if it meets the configured support and residual requirements.
To inspect a specific work surface, restrict the input with a mask or
region.

The scan and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
