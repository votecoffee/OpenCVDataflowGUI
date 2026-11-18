# Sources, licenses and transformations

**Powered by Poly Haven.**

**Charlotte Baglioni**, *Anti Slip Concrete*, Poly Haven:
[asset](https://polyhaven.com/a/anti_slip_concrete),
[publisher license](https://polyhaven.com/license),
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The primary author/asset and CC0 pages were rechecked on 2026-11-18.
The public API [terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md)
request an identifying User-Agent and lightweight visible credit. The original
fetcher identifies itself and this README retains that credit. No endorsement is
implied; publisher website screenshots and preview renders are not used.

The selected original linear diffuse EXR and uint16 OpenGL normal PNG remain
byte-for-byte in [sources](sources). [Provenance](sources/provenance.json) records
exact source URLs, release identity, publisher MD5 and verified SHA-256.
[Transform details](sources/transform.json) record linear green-channel albedo,
256×256 area resampling, renormalization and the OpenGL-to-camera normal mapping
`[Nx, -Ny, -Nz]`. Local directional diffuse-cosine rendering makes normalized
lighting stacks. Injected bright/dark disagreement and missing/ceiling samples
are generated controls. They are not acquired multi-light photographs, measured
lighting calibration or an optical/material qualification dataset.

Other derivatives include sealed reference/capture archives, source masks,
normal/albedo/support/reason maps, relighting and diagnostic PNGs. The recorded
capture source transports a separate runtime stack; it is neither an acquisition
device nor a fallback to training. Source-aligned flag images, outlines, labels
and gallery resizing are presentation only. The fixed generic fit settings and
95% availability decision remain unchanged; no test-image tuning or relabeling
was used. Newly staged models add the original-reference fit report as metadata.

Original analytic controls, example code and the tool use the supplied
[MIT license](LICENSE). No private images or private source-repository attribution
are included. Dependencies and upstream assets retain their own terms.

The small Strathclyde acquired-sphere dataset is **not included or used**.
Its publisher describes it as CC BY 4.0, but the ordinary download returned
HTTP 403 again on 2026-11-18. A larger Martine CC BY 4.0 dataset is a research
candidate only; no image/calibration subset has been downloaded or qualified.
No acquired-camera or physical normal/reflectance/height accuracy claim follows
from these rendered examples.
