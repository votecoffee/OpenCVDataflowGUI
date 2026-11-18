# Dataset attribution and transformations

**Michał Własiuk** (producer), **Robert Sitnik** (supervisor), Warsaw University
of Technology. *Simulated structured light 3D scanner datasets*, version 1.0.0,
Zenodo, DOI [10.5281/zenodo.17826191](https://doi.org/10.5281/zenodo.17826191).
The primary record/API licenses the data under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
No endorsement is implied.

Used files come from `camera-2592-1944-projector-640-480.zip`, the **perspective**
camera model and **crown** object. `sources` preserves six RGB PNG
observations, six source projector commands and small source-scene declarations.
`sources/provenance.json` pins exact member paths, sizes, CRC and SHA-256. The archive
size and publisher MD5 are retained; only selected ranges were downloaded and the
whole-archive MD5 was not independently verified. No renderer/model/code ran.

Changes for the phase-only example are explicit in `sources/transform.json`:

- Select the green channel and apply an sRGB inverse, based on the supplied PBRT
  RGB Film/PNG scene and its standard writer. This is a declared preprocessing
  inference, not a verified measurement of the exact renderer build. The original
  files have no encoding tag. The original RGB PNGs remain unchanged.
- Average 3×3 blocks after linearization, retaining the complete field as 864×648.
  Any original ceiling pixel propagates as an unavailable ceiling state. The
  source-from-derived pixel transform is retained; no depth is inferred from it.
- Six source projector commands establish a 6-pixel X period, +pi/3 phase increments
  and initial 2pi/3. This defines declared phase order independently of test results.
- Fit using unchanged generic phase settings. Retain amplitude, relative RMS,
  unavailable reasons and wrapped phase. No camera/projector reconstruction
  calibration or physical shape reference is imported for the crown.

This is an **external simulated phase-evidence example**, not real acquisition,
reconstruction ground truth or a qualified metric scan. About 10.18% of the field
passes the initial generic phase gates; the low coverage and poor fits remain
visible. Dataset input selection and transformations do not establish accuracy.

The sRGB inference uses the primary
[PBRT image-format implementation](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/image.cpp)
and [PBRT image/encoding reference](https://pbr-book.org/4ed/Utilities/Images).
No PBRT implementation code is copied into this tool.

The two tilted-plane examples are original analytic software controls. Their
retained XYZ is independently constructed from camera rays and a known plane,
with one visible surface per camera ray. They are not physical gauge evidence.
Original code/controls use [MIT](LICENSE). Dependencies retain their terms:
NumPy BSD-3-Clause, OpenCV Apache-2.0, Pillow MIT-CMU. No neural model is included.

License rechecked against the primary Zenodo API: metadata.license.id = cc-by-4.0.
Additional controls shift X phase alone by 2px, or coherent Y Gray+phase by 2px
in a fixed source rectangle; acceptance settings are unchanged. Zero-source and
graphics-off controls preserve unavailable-state evidence. No new image download
or third-party renderer execution was required.
