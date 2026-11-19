# Train a HOG sliding-window detector

Use **Train HOG detector with hard negatives** to build an object detector from
labeled image crops. Batch positive and negative crops separately, then connect
them to the trainer. Each class needs at least two crops with the exact
configured window geometry. The trainer takes uint8 gray, BGR or RGB images and
converts them to the stored gray or canonical BGR encoding.

Connect one or more separate target-free frames to **Backgrounds** for one
hard-negative mining pass. Use backgrounds that show likely distractors and keep
your validation scenes separate. The trainer checks duplicate working pixels
within and across the supplied groups. The operator chooses which scenes are
target-free and which captures belong in each split.

The block computes native HOG descriptors, fits a binary OpenCV linear C-SVC,
and verifies that the exported target-positive margins agree with the native
SVM. It scans the full mining frames at the original detector window scale,
using the configured stride and margin threshold. The retained windows are
ordered by descending margin with stable image and coordinate ties. Descriptors
are computed at those locations in the original frame, preserving image context
at window boundaries. A second fit includes the retained distinct negatives.
The trainer exposes the initial and refitted detector vectors. [OpenCV HOG reference](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html)

The report records crop and background hashes, each retained source location and
baseline margin, native/vector parity, and candidate counts on the mining frames
before and after training. The search computes descriptors for at most four times
the requested negative count while eliminating duplicates. If no distinct
candidates are retained, Model and Baseline model carry the same revision.

The tool caps input rows, pixels, windows and estimated feature work before
native fitting. Training accepts up to 512 total labeled crops and 32 background
frames, subject to the tighter work limits. This mining pass uses one scale.
The existing **HOG sliding-window detection** block can run the returned model
over multiple scales.

Connect Model to **Save HOG detector model** when you want a reusable file.
The saved model carries exact HOG geometry, class name, weights and bias,
version and training identity. **Load HOG detector model** verifies the revision
and feeds the regular detector. Use a common hit threshold and suppression
policy when comparing the baseline and final detector on held-out scenes.
Margins are signed classifier scores. Set production thresholds and measure
recall and false positives using representative independent captures.

[Training and held-out comparison projects](<../examples/opencv tools/hog_detector_training/README.md>)
