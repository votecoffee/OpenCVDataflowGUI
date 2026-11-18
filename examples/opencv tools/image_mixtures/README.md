# Color and texture Gaussian mixtures

Open a project and choose **Run all** or **Run once**. Each project uses one image;
Run all falls back to one execution. Images, truth masks and models are included.
See [the block contract](../../../docs/image-mixtures.md) and [sources](SOURCES.md).

| Project | Purpose |
| --- | --- |
| [train_color](train_color.json) | Train selected color distributions and save a new model |
| [train_texture](train_texture.json) | Train local mean/std features inside two selected surfaces |
| [color_known](color_known.json) | Reuse the model on independent noise; expected pass |
| [color_defect](color_defect.json) | Find exactly 120 unfamiliar pixels; expected fail |
| [density_disabled](density_disabled.json) | Same defect with density disabled; deliberately passes |
| [component_filter](component_filter.json) | Accept only canonical component 0; expected fail |
| [region_color](region_color.json) | Restrict inspection to a right-hand region; expected pass |
| [color_rgb16](color_rgb16.json) | Reuse fixed normalized features on RGB uint16 input |
| [spherical_color](spherical_color.json) | Spherical fitting with equivalent diagonal prediction |
| [posterior_maps](posterior_maps.json) | Expose the optional group of component probability maps |
| [texture_known](texture_known.json) | Supported neighborhoods on held-out texture; expected pass |
| [texture_defect](texture_defect.json) | Reduced texture amplitude; expected fail with boundary misses |
| [texture_support](texture_support.json) | Deliberate 100% support requirement; expected fail |

Training writes `trained-mixture.json` beside the project and preserves existing
files. Choose a fresh destination to repeat training. Query projects load their
frozen model from assets; no plugin, download or training framework is needed.

The color edit has exactly **120 pixels** and is found without extras/misses.
Its winning component probability is **1**, showing why relative probability
alone cannot establish novelty. Density and optional component/support gates
control the inspection. Default maps retain the original image and current ticket;
Debug adds drawable filmstrip outlines and a diagnostic component map. Lean keeps
the same numeric result without those graphics.

Texture uses 7×7 neighborhoods fully inside both regions. There are **3,400 valid
centers** from **4,480 selected pixels**. The chosen threshold flags **192 of 256**
edited pixels, misses **64**, and adds none. Neighborhood mixing at boundaries
limits localization; this is not a production defect-sensitivity benchmark.

Run `benchmark.py` to compare independent window statistics, native per-pixel
predictions and Gaussian formulas against the edit masks. Recorded results are in
[benchmark.json](benchmark.json). `build_example.py` regenerates the original assets,
frozen models and graphs and preserves an existing assets folder. The models store
explicit covariance regularization and prediction representation; see the contract
for spherical/native-initialization details and limits.
