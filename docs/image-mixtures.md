# Image surface Gaussian mixtures

Use **Train image Gaussian mixture** (`ml.em_train_image`) to learn the distribution
of selected surface colors or local texture. Save the model with `ml.em_save`,
load it with `ml.em_load`, and connect it and a current image to
**Classify image Gaussian mixture** (`ml.em_classify_image`). The
[thirteen projects](<../examples/opencv tools/image_mixtures/README.md>) run offline
with original images, truth masks and included models. No extra package is needed.
These are unsupervised components, not externally labeled semantic classes.

## Features and fitting

Feature spaces are normalized BGR, Lab (L/100 and a,b/128), gray, or grayscale
neighborhood mean/standard deviation. Integer uint8/uint16 values use their fixed
full-scale range; floats must already be in [0,1]. RGB and alpha encodings are
interpreted explicitly. There is no per-image normalization or white balancing.
Gray inputs retain their direct normalized samples instead of making a round trip
through color. The original image pixels, depth, encoding and alpha remain unchanged.

Optional masks and rectangle/circle/polygon regions intersect. Alpha-zero pixels
are excluded. Texture uses an odd 3–63 pixel square and only publishes centers
whose entire neighborhood is visible and selected, including at image borders.
Features cannot therefore read across a selection hole. The Valid mask and support
fraction expose that erosion; unsupported pixels are not counted as unknown.
Empty effective support is an explicit configuration/input error.

Native EM fitting exposes 1–16 components, diagonal/spherical/full generic
covariance, count-plus-epsilon stopping, sampling and initialization controls.
Auto uses native `trainEM` with its k-means initialization. Manual converts the
supplied initial means, optional covariances/weights and default initial variance
into Gaussian responsibilities, then starts native `trainM`. It is not `trainE`:
the installed Python binding did not accept the multidimensional covariance
representations tested. Manual initialization is deterministic for these fixtures
and does not change the global OpenCV random seed. Auto fitting may vary.

Training samples supported pixels evenly in row-major order, up to Max samples.
Rare surfaces can be missed. Fitted components are sorted lexicographically by
their means; IDs are stable within a saved model, not semantic identities across
retraining. Termination does not establish optimality or convergence quality.
An explicit covariance eigenvalue floor is applied after fitting in squared
feature units. The portable native model stores the effective covariances and
the report retains original minimum eigenvalues, fitting method and counts.

Spherical fitting retains the native spherical fitter, but the stored prediction
model uses the equivalent diagonal representation of each scalar-times-identity
covariance. In this installed OpenCV 5 build, direct multidimensional spherical
scores differ from the ordinary Gaussian normalizer. Canonical diagonal scoring
agrees with the independent Gaussian reference; fitted and prediction covariance
types are both reported. This does not repair the spherical fitting optimizer.
The representation choice and regularization are fixed model transformations,
not silent runtime retries or changes to global OpenCV settings.

## Classification and judgment

Labels are candidate component IDs, even when rejected; -1 marks unsupported pixels.
Winning posterior is the native maximum conditional component probability, with -1
outside support. Optional Posteriors is a hidden-by-default group of per-component
float images, usable with existing Select/Unbatch tools. Unsupported values in
those maps and Log likelihood are zero; use Valid mask to distinguish them.

A posterior compares only the trained components. An unfamiliar pixel can have
posterior **1**. An enabled absolute log-density floor, minimum posterior and
optional accepted-component IDs jointly define Accepted mask and Unknown mask.
Density uses a bounded vectorized Gaussian calculation on the exact saved native
parameters; independent per-pixel native `predict2` checks agree. Density depends on
feature units and dimension and is not a probability. Do not transfer thresholds
between feature spaces without validation.

The block passes when the unknown fraction among supported pixels is at or below
its inclusive limit and supported/requested fraction meets its minimum. Score is
the accepted-pixel fraction, not confidence in correctness. Fail reasons distinguish
unknown features from insufficient support. The block does not assign a final
ticket judgment. Current image, numeric maps and report remain available on a valid
inspection failure. Bounded drawable unknown contours and a status label reach the
filmstrip; a diagnostic component map is retained when graphics are enabled.
Lean skips those graphics while preserving numeric decisions.

All current maps retain source ticket, spatial/pyramid, recipe and companion data.
Missing wired selections and failed-input policies gate work before last-input
history changes. Native handles are local to an invocation; no trained handle is
serialized or shared across copies. Native exceptions are contained by the common
executor and later valid input can recover.

## Persistence, bounds and evidence

Models use strict JSON with duplicate/nonfinite rejection, a 64 KiB file limit and
content revision. Matrix dimensions, weights, covariance structure/positive
definiteness, feature/OpenCV version and training counts are checked before native
read. A revision verifies content consistency, not publisher authenticity.
Save preserves existing destinations; project Save As copies sealed model bytes.
JSON/XML reopen uses those same parameters without retraining. Work, image size,
sampling, prediction chunks and estimated workspace are bounded. Workspace limits
are estimates, not measured process RSS; optional posterior maps need more memory.

The original controls use separate training/query noise and excluded held-out pixels.
The color fixture finds the exact **120-pixel** edit with no extra/missed pixels,
despite winning posterior **1** throughout. Disabling density deliberately accepts
that edit. Known color and texture controls have no unknown pixels. Native scores
and independent Gaussian calculations agree within **2.3e-13** on the benchmark.

The texture fixture has **3,400 supported centers** from **4,480 selected pixels**.
At its chosen threshold it flags **192 of 256 edited pixels**, with **64 misses**
and no extras. Window mixing suppresses boundary responses. Those decisions describe
feature neighborhoods and do not claim exact defect boundaries or production
sensitivity. Fit populations, feature scale/window and thresholds need separate
training, tuning and validation images for each application.

Primary references: [OpenCV EM API](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ml_1_1EM.html)
and [native implementation](https://github.com/opencv/opencv_contrib/blob/5.x/modules/ml/src/em.cpp).
