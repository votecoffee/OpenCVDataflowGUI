# Sources and license

Images, masks, graphs, trained models and code are original work by
OpenCVDataflowGUI contributors under the repository's [MIT license](../../../LICENSE).

`build_example.py` creates 64×96 images with two noisy color surfaces or two
equal-mean surfaces with different texture amplitude. Separate deterministic seeds
produce training and query pixels; query and edit images are excluded from fitting.
The RGB16 demonstration stores the same normalized query values as uint16 and
uses the loader's explicit RGB conversion. The color edit is a 10×12 rectangle;
the texture edit is a 16×16 rectangle. Independent binary masks record those edits,
not the detector output or its neighborhood support.

`truth.json` records all image/model SHA-256 values, case expectations and region
geometry. Native training uses manual Gaussian responsibilities and `trainM` for
the frozen models. Covariance flooring and canonical component ordering are
explicit; spherical prediction uses an equivalent diagonal representation.
`benchmark.py` imports no application implementation: it computes features
independently, reads saved parameters into native EM, calls per-pixel `predict2`
and compares scores to a Gaussian formula and detections to the edit masks.

The synthetic controls validate software contracts and expose known misses;
they do not establish real-surface accuracy, radiometric calibration or speed.
