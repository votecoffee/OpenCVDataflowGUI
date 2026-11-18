# Sources, license and scope

**Powered by Poly Haven.**

**Charlotte Baglioni**, *Anti Slip Concrete*, Poly Haven:
[asset page](https://polyhaven.com/a/anti_slip_concrete),
[asset license](https://polyhaven.com/license),
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The live publisher asset page/author and CC0 license were rechecked 2026-11-17.
The public API [terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md)
also request an identifying application User-Agent and lightweight visible credit;
the original fetcher identifies itself and the example README displays that credit.
No endorsement is implied. Publisher website screenshots/preview renders are not used.

Original selected 1k linear diffuse EXR and uint16 OpenGL normal PNG files are retained
byte-for-byte in [sources](sources). [Provenance](sources/provenance.json) records
exact public file URLs, release hash, publisher MD5 and locally verified SHA-256.
The release was pinned before fitting. [Transform details](sources/transform.json)
record decoding, resampling and coordinate mapping.

Derivatives: local diffuse-cosine rendered lighting stacks, normalized arrays, source
masks, frozen recipes, maps and diagnostic views. They are **rendered controls** using
material-map inputs, not acquired camera data, measured light calibration, a complete
optical simulation or physical normal/albedo accuracy evidence. Source material maps
provide the declared rendering reference; injected highlights/darkness/missing rays
are synthetic observations. No capture-dependent settings are learned from test data.
Original analytic control data and code use the supplied MIT license.

The known-lighting recipe/settings are unchanged across the original five captures;
the two contaminated captures reuse their clean training models. Additional cases
exercise connected transport, declared exposure compensation, completely missing
input and optional execution graphics without tuning fit thresholds. The recorded
stack source is an offline array transport, not an acquisition device or training
fallback. Its capture archive is separate from the execution model.

Completed reviews preserve actual normalized native codes, masks, lighting/exposure
declarations, prepared observations and stored maps. Original encoded source-image
bytes are outside this runtime capture contract. Estimated normals are directions,
not measured XYZ or metric height. Relative albedo, fit residuals and validity do not
identify physical material or defects. Source-aligned display scaling/markers and
observation-minus-residual predictions are derived presentation, not new estimation.

Acquired-stack registration/response/lighting calibration, real material behavior,
independent measurement uncertainty and broader material qualification remain open.
The tool and example code preserve the original MIT copyright in [LICENSE](LICENSE).
No private images or private repository attribution are included.
