# Used data and dependency credits

**Powered by Poly Haven.**

**Charlotte Baglioni**, *Anti Slip Concrete*, Poly Haven:
[asset page](https://polyhaven.com/a/anti_slip_concrete).
Poly Haven releases its assets under
[CC0](https://polyhaven.com/license). The public API's
[terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md) require an
identifying application User-Agent and a visible lightweight credit. The
fetcher identifies itself and the example overview/README show that credit.
No endorsement is implied.

The selected 1k linear diffuse EXR and uint16 OpenGL normal PNG are retained
byte-for-byte under `studies/polyhaven_source`. Publisher MD5 and local SHA-256
hashes, exact public file URLs and the asset release hash are recorded in
`provenance.json`. The release was pinned before fitting. Their normalization,
resampling and coordinate mapping are recorded in `transform.json`.

The derived light stacks, output maps and overview images are **rendered
diffuse controls** using these material maps. They are not acquired camera
stacks or evidence of real material reflectance/physical shape accuracy. The
reference normals/albedo test the declared rendering construction. The
original analytic controls and example code use this package's MIT license.

Dependencies retain upstream terms: NumPy BSD-3-Clause; OpenCV Apache-2.0 for
the supported releases; Pillow MIT-CMU. Optional example curation uses the
[OpenEXR Python library](https://openexr.com/en/latest/python.html), BSD-3-Clause.
Only original code is packaged in the tool wheel; external libraries retain
the notices supplied by their distributors. No neural weights are included.
