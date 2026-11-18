# Visual-word image retrieval

Build a reusable image-retrieval pipeline with **Visual local features**
(`ml.visual_features`), **Train visual words** (`ml.bow_train`),
**Image visual-word histogram** (`ml.bow_encode`), **Build visual-word gallery**
(`ml.bow_gallery`) and **Rank visual-word images** (`ml.bow_rank`). Save/load
codebooks and galleries with `ml.bow_save`/`ml.bow_load`. These tools appear under
**OpenCV → ml → Image retrieval**. The
[twelve projects](<../examples/opencv tools/visual_words/README.md>) run offline
without additional packages or downloads.

## Extract and train

The extractor uses native SIFT float32 descriptors with Euclidean distance, or
native ORB bytes with Hamming distance. ORB WTA 3/4 uses Hamming2, which counts
different two-bit groups. Binary words remain actual sampled descriptors;
averaging their byte values would change the metric's meaning.

All relevant detector controls are editable: retained feature count, SIFT octave
layers/contrast/edge/sigma/precise upscale, and ORB scale/levels/border/first level,
WTA, score mode, patch size and FAST threshold. The SIFT descriptor representation
is fixed to native float32. Unused settings for the other detector do not enter
the active preprocessing identity. The identity includes the OpenCV version.
Points are sorted by response and geometry and capped at Max features.

Images convert to uint8 gray using fixed depth scaling; floating inputs must be
in [0,1]. The original image keeps its pixels, encoding and depth. Optional mask
and rectangle/circle/polygon region selections intersect; transparent pixels are
excluded. Selection limits keypoint centers. Descriptor patches can still read
neighboring pixels outside the selected region. This is not a strict descriptor
support mask. Missing wired selections suppress work through the common runtime.

Training accepts one feature record or a batch of up to 64 compatible records.
Even row sampling first caps each image, then caps the combined population.
Farthest-first selection initializes distinct words deterministically. For SIFT,
native k-means refines those centers with one initialized attempt and configurable
iteration/epsilon stopping. For ORB, the selected binary descriptors are the
finished farthest-first medoid vocabulary; no k-means or medoid refinement is
claimed. Sorted stored words define stable IDs within that codebook. Fewer
distinct descriptors than words is an explicit error. Blank training populations
are rejected. Selection/sampling and the training set affect the resulting model.

## Encode, store and rank

Encoding checks the feature record and packet identity against the codebook,
then assigns each descriptor using native brute-force matching. Optional Max word
distance rejects assignments above an inclusive native-unit threshold; -1 disables
rejection. Rejected word IDs are -1. Outputs include all assignment distances,
IDs, accepted count and a normalized image signature. L1 gives word frequencies;
L2 gives a unit vector; Hellinger takes the square root of L1 frequencies.

No features or no accepted assignments produces a valid inspection failure with
a zero signature. A gallery refuses zero signatures. An empty query ranks nothing
and fails the minimum-result check; it does not produce an arbitrary nearest
image. The signature's contract includes codebook revision, normalization and
assignment threshold. Gallery building and ranking require exact matching
contracts, even when two codebooks have the same number of words.

Batch signatures and build a gallery with explicit labels or the batch's labels.
Multiple views may share a label. Stored entries retain reference ticket,
coordinate, pyramid and recipe context separately from the current query.
Ranking returns sorted entry indices, labels, distances and reference context.
Exact ties keep gallery order. Choose cosine, L2, alternative chi-square or
Bhattacharyya distance; distance limits are metric-specific and inclusive.
Top k and Minimum results determine the valid inspection judgment. Distances
are not confidence probabilities and no final ticket judgment is assigned.

Optional review images must agree with known current ticket, frame, pyramid and
recipe context. Outputs preserve their pixels and metadata. Drawable feature
centers and retrieval labels can reach the filmstrip through existing overlays;
the histogram is also available as a diagnostic plot. Lean omits optional graphics
without changing numerical results. Native exceptions are contained by the base
executor and suppress current emissions; later valid inputs can recover.

## Persistence and resource limits

Codebooks store exact native descriptor/center values, extractor settings, metric,
training counts and a content revision. Galleries store signature identity,
entries and their own revision. Strict JSON rejects duplicates, nonfinite values,
unknown schema fields, invalid norms/coordinates/metrics and files over 8 MiB.
Revisions check content consistency, not publisher authenticity. Native detector
and matcher handles are invocation-local. Save preserves existing destinations;
sealed load assets survive Save As and JSON/XML reopening without retraining.

Extraction supports up to one million pixels, 16,384 pixels per dimension and
4,096 retained descriptors. ORB rejects a pyramid above 16 million estimated
pixels before detector creation, including extreme first-level upscaling.
Training accepts at most 32,768 input descriptors and samples at most 8,192;
the vocabulary has 2–128 words. Default training samples at most 512 per image
and 2,048 overall. Training/encoding expose work and workspace limits, defaulting
to 128 million estimated operations and 128 MiB. These estimates are not process
RSS limits. Building currently uses the runtime's 64-entry batch limit; portable
galleries can hold up to 1,024 entries. Ranking is exact and bounded by those
gallery/signature limits. This is a small-gallery baseline, not an approximate
index or a large-scale retrieval benchmark.

## Controlled evidence and limits

Six original MIT training images use three repeated part-like motifs with ±3°
rotation and independent noise. Three held-out queries use 7° rotation, separate
noise and gain 0.95; their pixels/hashes differ from training. All three queries
rank the intended gallery label first with SIFT and ORB. Best cosine distances
are **0.0119/0.0321/0.0396** for SIFT and **0.0057/0.0158/0.0871** for ORB.
Hellinger signatures also retain the expected first ranks. Independent native
extraction and NumPy quantization/ranking agree on every assignment in the
recorded controls; SIFT distance roundoff is below **2.8e-5**, ORB distances agree
exactly, and the complete saved pipelines match the independent signatures.

A shuffled-quadrant query still ranks as washers, at SIFT distance **0.0087**.
This exposes the loss of spatial layout in a bag of local descriptors. The tools
retrieve similar appearance; they do not verify geometry, part count, pose,
occlusion or pickability. An intentionally strict distance threshold rejects even
a known query. Real application thresholds, illumination/scale variation and
unknown-image rejection require separate validation, rather than extrapolating
accuracy from these controls.

Primary reference: [OpenCV 5 visual-word descriptor](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1BOWImgDescriptorExtractor.html).
The blocks compose native extraction, k-means and matching with explicit portable
contracts; they do not call the image-only BOW helper or require a native handle
to persist between runs.
