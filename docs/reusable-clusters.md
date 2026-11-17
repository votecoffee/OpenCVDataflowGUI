# Reusable image clusters

The **OpenCV → Machine Learning → Clustering** tools train color palettes or
image-descriptor vocabularies once, then reuse their centers on separate images.
Existing **K-means image pixels** and **K-means samples** still fit each invocation.
They are useful for exploration; their IDs are not a saved classification model.

| Block | Purpose |
| --- | --- |
| Train image color palette (`ml.palette_train`) | Fit selected visible pixels from a reference image. |
| Apply image color palette (`ml.palette_predict`) | Label another image without refitting. Produce distances, unknown mask, histogram and quantized pixels. |
| Train descriptor clusters (`ml.kmeans_train`) | Fit numeric feature rows or a batch of image descriptors, such as HOG or embeddings. |
| Predict descriptor clusters (`ml.kmeans_predict`) | Assign rows to the stored nearest centers and produce a visual-word histogram. |
| Save cluster model (`ml.kmeans_save`) | Write a new portable JSON model, preserving existing files. |
| Load cluster model (`ml.kmeans_load`) | Validate and emit a saved palette or descriptor vocabulary. |

Use **Image file → Train palette → Save model** to train, then **Load model +
Image file → Apply palette** to inspect other captures. For descriptors use
**Image files → HOG → Batch serial entries → Train clusters → Save model**.
Prediction takes the matching descriptor preprocessing and the saved model.
Recipes may select another model through its input or loader path.

## Training and stable IDs

Training exposes cluster count (1..32), random or K-means++ initialization,
attempts, termination mode, iteration limit and center-movement epsilon. The
descriptor trainer accepts optional initial labels for the first attempt and
optional per-row L2 normalization. A zero vector cannot use L2. Native fitting
uses float32. Attempts select the smallest native sum of squared distances.

Centers are ordered lexicographically once at training; their array positions
are the persistent zero-based IDs. Prediction never sorts or updates them.
An exact nearest-center tie chooses the lowest stored ID. IDs have no automatic
semantic meaning. Retraining can change centers and their IDs; retain the model
revision with any interpretation of them. Native random initialization is not
promised to reproduce a retraining run. Training does not alter global RNG seeds
and does not opt into parallel branch execution. Stateless predictors may run in
independent branches and do not share native model handles.

Descriptor models retain the `feature_contract` supplied by the input. Contracts
within a training batch must agree. Prediction defaults to exact contract matching;
the explicit **Dimensions only** option delegates preprocessing compatibility to
the caller. Saved normalization always applies. HOG supplies its native version,
geometry, encoding, depth and controls. The trainer also accepts ordinary numeric
rows without a contract; their preprocessing consistency is the caller's job.

## Color preprocessing and selections

Palettes accept tagged gray, BGR, RGB, BGRA or RGBA images with uint8/uint16 depth,
or finite float32/float64 samples in [0,1]. Integers use their full-scale maximum;
there is no per-image stretching or automatic white balance. Equivalent integer
depths and RGB/BGR encodings share the same working representation.

Choose normalized BGR, gray luminance, or native float Lab scaled by
`[100,128,128]`. Lab distance is Euclidean in that scaled space, **not Delta E**.
The model records the working space and exact OpenCV version; a different color
preprocessing version must be explicitly retrained/migrated. Convert HSV/Lab-tagged
input images to a supported source encoding first. Physical depth planes use the
separate depth tools.

Masks and rectangles/circles/polygons intersect. Alpha-zero pixels are excluded;
other alpha values are not weights. Training samples up to **Max samples** evenly
from the selected pixels in row-major order. This deterministic sampling can miss
rare colors: raise the limit or select representative regions. The report records
selected and fitted counts. Prediction evaluates every selected visible pixel.
An empty selection is a configuration/input error with no output event. Connected
selections must arrive; a missing region never falls back to the full image.

The primary Image output preserves original pixels, encoding, depth and metadata.
Quantized is a separate numerical image using the learned center colors; selected
pixels change, while outside pixels and original alpha remain exact. Lab conversion
and integer rounding can introduce small reconstruction errors. Applying a color
palette to gray pixels produces luminance of the assigned center for quantization.
Labels, distances and mask outputs use the current image's grid, ticket, recipe
and pyramid context. They do not inherit the training image's ticket.

## Unknown samples and graphics

**Max distance** is an inclusive Euclidean cutoff in the stored working feature
space; zero disables it. **Accepted clusters** optionally restricts IDs. A sample
is unknown if either filter rejects it. Distances are not probabilities.
**Max unknown fraction** allows some unknown samples before the tool fails.
Equality passes. A valid inspection failure still emits its measurements and
`palette_unknown_pixels` or `cluster_unknown_samples` reason; it is not a final
ticket judgment. Invalid models/settings or native exceptions emit no results
and use the common diagnostic logger.

Palette Labels is int16, with -1 outside the selection and the nearest stored ID
at every selected pixel, including unknown pixels. Unknown mask independently
flags rejected pixels as 255. Distances is float32, with -1 outside the selection.
Descriptor prediction returns one ID, distance and acceptance flag per row.
Histogram contains accepted counts divided by **all selected samples**; its sum
can be less than one. Unknown count/fraction and raw accepted counts are separate
report fields. Histogram payloads carry a feature contract including the model
revision and rejection controls, for downstream classifiers.

Palette result graphics are drawable unknown outlines and a summary label.
Connect Result graphics and Image to **Overlay result graphics**, then to
**Filmstrip image**. Cluster map is a separate diagnostic raster; red means
unknown. Lean skips optional drawing/cluster-map work. Primary measurements,
unknown masks and numerical quantization remain available. Drawings are bounded
to 128 contours/8,192 vertices; the full mask remains authoritative.

## Persistence and limits

Models contain centers, normalization, feature contract, kind and a SHA-256
revision. No pickle, native handles or executable content is stored. The checksum
detects modifications, not authorship. Loader JSON rejects duplicate keys,
nonfinite numbers, changed revisions, malformed centers and incompatible color
contracts. Centers retain exact float32 values. Load paths are sealed project
assets and copy on Save As; JSON/XML preserve all controls.

Images are limited to one million pixels. Descriptor tables allow 100,000 rows,
65,536 columns and two million values, finite within ±1e15 and without float32
underflow. Center tables allow 32 rows/131,072 values; models are bounded to
8 MiB and feature contracts to 16 KiB. Prediction allows 64 million
row × center × feature comparisons and uses bounded float64 direct differences
to avoid cancellation near large coordinates. Use a smaller pyramid level or
fewer clusters for larger workloads. There is no automatic metric calibration,
spatially aware clustering, multi-reference palette merging, incremental fitting
or calibrated unknown-class confidence in this baseline.

See [nine runnable projects](<../examples/opencv tools/reusable_clusters/README.md>).
Native fit reference: [OpenCV clustering](https://docs.opencv.org/4.x/d5/d38/group__core__cluster.html).
