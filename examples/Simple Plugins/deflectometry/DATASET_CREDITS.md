# Dataset credits and changes

## OpenCSP SOFAST Landscape NSTTF facet

Courtesy **OpenCSP**. **OpenCSP Team**, *OpenCSP: An Environment for Collaborative
CSP Optical Technology Development*, Sandia National Laboratories, National Solar
Thermal Test Facility (NSTTF), [official project](https://opencsp.sandia.gov/).

Source: [NSTTF_Facet_SOFAST_Landscape.zip](https://sandia-csp.app.box.com/s/iftmhdkhgjnmfgefsfmkid011dtv8n3g/file/1951928463619),
83,154,541 bytes, publisher SHA1 `61485298ea4fda78425d0e70f0d59c87bc9a58aa`, local
SHA256 `ce027803717c3e97cc4e4481b53e7ac6a90e822c151d1c850f4023233e24b5ff`.
The publisher SHA1 and selected ZIP CRC/member SHA256 values were verified.

**Data license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).**
The [official separate data license](https://sandia-csp.app.box.com/s/iftmhdkhgjnmfgefsfmkid011dtv8n3g/file/1341437503062)
applies to the directory tree. Its original 145 bytes are retained as
[OpenCSP_Box_license.txt](sofast_source/originals/OpenCSP_Box_license.txt), SHA1
`63b5eeee74062d1ea1fb194bfcce0a81afe58f74`. The original code license is separate
and does not replace data terms. Original tool code and generated analytic
controls use the tool's MIT license. The retained SOFAST images/mask, transformed
capture and derived SOFAST result maps/review images retain CC BY-SA 4.0.

Retained originals: four coarse-frequency exported X camera PNGs, four Y camera
PNGs, the corresponding eight command-thumbnail PNGs, two dark/white mask-source
PNGs and the archive README. Each member has pinned hashes in
[provenance.json](sofast_source/provenance.json). The archive itself and the
processed 67,893,704-byte HDF5 stay outside the published example. No upstream
renderer, capture controller, model or executable script ran.

Changes: normalize 8-bit camera codes by 255; average each 2x2 code block from
1626x1236 to 813x618; propagate any source 255 ceiling into the derived pixel.
This is an explicit **code-value** reduction, not calibrated linear radiometry.
The original-to-derived pixel-center mapping is `x=2*u+0.5`, `y=2*v+0.5`.
The upstream HDF5 `CalculationImageProcessingGeneral/mask_raw` is exported as a
numeric mask and reduced with all four source pixels required. It defines the
inspection region; it is not a defect label or independently measured ground truth.

The first four exported X/Y frames are assigned shifts `[0,-pi/2,-pi,-3*pi/2]`,
consistent with the upstream [fringe generator](https://github.com/sandialabs/OpenCSP/blob/main/opencsp/app/sofast/lib/Fringes.py)
and [image export](https://github.com/sandialabs/OpenCSP/blob/main/example/sofast_fringe/example_process_single_facet.py).
An independent four-step check of the retained command thumbnails gives maximum
fit error 0.25 byte code and about -0.899832 cycles across each thumbnail. Physical
screen pixel pitch/period is not inferred from those resized command illustrations;
unknown physical period/origin remain null in the capture manifest.

The real capture is compared **with itself** in `sofast_self_review`. Exposure,
response calibration and independent measured shape are unavailable; the report
records uncalibrated-code diagnostics, null exposure and no metric height. A
successful self-comparison establishes import/fit/replay behavior only. It is
not evidence of held-out defect detection, repeatability or physical accuracy.

## Generated controls

The analytic clean/change/missing captures are independently constructed
sinusoidal phase fields with declared shifts and direct phase-change truth.
The change control modifies a localized phase field; it is not a measured 3D
deformation. Missing/clipped samples remain invalid. Known reflection-normal
and half-angle truth are checked separately by the numerical tests.
