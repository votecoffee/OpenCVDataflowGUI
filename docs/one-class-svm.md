# One-class image anomaly inspection

Train an OpenCV one-class RBF SVM from known-good image descriptors. Connect a
consistent descriptor block such as **Histogram of oriented gradients** to
**Batch serial entries**, then connect the group to **Train one-class SVM on
known-good images**. Select the same crop, resize, image format and descriptor
settings for all training and query images. The trainer needs at least three
distinct descriptors after normalization.

Tune nu and the RBF kernel scale gamma on representative data. The training
report shows the number accepted at the native boundary and at the configured
cutoff. Use separately acquired known-good and defective images to measure
false rejects and missed defects. The one-class training method estimates a
boundary from one population. [OpenCV SVM reference](https://docs.opencv.org/4.x/d1/d2d/classcv_1_1ml_1_1SVM.html)

Connect Model to **Save one-class SVM model** and choose a new JSON filename.
Connect **Load one-class SVM model** to **Inspect image using one-class SVM**.
A model placed under the project directory travels with Save As. The loader
checks the revision and reconstructs the native model from bounded numeric support
vectors, weights and bias. The stored revision covers the preprocessing contract,
normalization, training descriptor hash, solver settings and decision cutoff.

The inspector returns the native raw margin. The OpenCV 5.0.0 path was verified
against native labels and an independent RBF calculation on numeric and held-out
image controls. Positive raw margins correspond to native inliers. The reported
anomaly score negates that margin, so higher values indicate greater anomaly
evidence for the same model. This score has no probability interpretation.

Accept only when raw margin is strictly greater than the saved minimum inlier
margin. A zero cutoff uses the native boundary. A positive cutoff can reject
a native inlier. The report exposes both decisions and the exact model revision.
A rejection includes the svm.anomaly reason and preserves the inspection ticket.
Connect the same source image for an unchanged review image with a drawn verdict.
This tool scores a whole descriptor. Use the local anomaly tools when you need
pixel locations.

Descriptor metadata is compared when a feature contract is present. Untagged
training and query rows are supported with dimensions and saved normalization
only. Add meaningful feature provenance to reusable image pipelines. Models
accept at most 512 training rows, 4,096 dimensions, and 524,288 descriptor
elements. The JSON file is limited to 16 MiB.

[Known-good training and held-out examples](<../examples/opencv tools/one_class_svm/README.md>)
