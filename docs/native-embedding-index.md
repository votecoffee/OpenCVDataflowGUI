# Native image embedding retrieval

**OpenCV > Features Framework (features) > Image retrieval** provides
`ml.native_ann_build`, `ml.native_ann_search`, `ml.native_ann_save` and
`ml.native_ann_load`. They use OpenCV 5's built-in `ANNIndex`, with no separate
Annoy Python package. Existing Annoy tools and their saved projects remain
available with their original IDs and formats.

## Gallery and query contracts

Build accepts image-derived descriptor rows or a batch of descriptors. Supply
unique stable Item IDs, such as image filenames, and optional display/class
Labels. Blank IDs use batch member labels or sequential strings. Labels may
repeat; IDs may not. Native contiguous row indices are internal mappings, and
results retain both row index and stable image ID. Reordering a gallery does not
rename its images. Descriptor normalization and preprocessing contracts are
stored with the model. Conflicting batch contracts are rejected.

HOG/DNN embeddings can connect to Samples. **Compute image hash** records can
also connect directly, individually or as a batch; method, options and bit count
become a checked feature contract. No manual hex conversion is required. Plain
numeric descriptors without a feature contract are supported; the caller then
owns preprocessing consistency. Require match is the default query contract
policy; Dimensions only explicitly relaxes that guard while retaining saved
normalization. Known ticket, recipe, pyramid and frame conflicts between a query
and its optional review image are rejected before native search.

Query accepts one descriptor or multiple rows. An optional Image accompanies
exactly one query and retains original pixels/depth/encoding/ticket. Result
graphics add the nearest label for overlay/filmstrip. Graphics-off skips drawings.
Reference gallery tickets do not replace current query provenance.

## Metrics and approximate candidates

| Metric | Reported measure | Better match |
| --- | --- | --- |
| Euclidean | L2 distance, not squared L2 | Smaller |
| Manhattan | Sum of absolute differences | Smaller |
| Angular | `sqrt(2-2*cosine)` | Smaller |
| Hamming | Differing bits in packed byte descriptors | Smaller |
| Dot | Inner-product similarity | Larger |

Search requests a bounded candidate count and explicit native `search_k`, then
reranks only those candidates using the exact metric on saved descriptors.
Ties in that returned set use stable Item ID order. Increasing candidates/effort
can improve recall; reranking does not recover candidates missed by the native
forest. Asking for all gallery rows with sufficient effort reproduces the exact
references in the included controls. This is not a guarantee of exact global
search for arbitrary data or effort settings.

Native result buffers start with missing-ID/NaN sentinels because the API may
return fewer candidates than requested. Only written, finite, valid, distinct
candidate rows are consumed. Missing candidates are counted in Report. Native
distances remain alongside reranked metrics; for Hamming the native distance is
the Euclidean distance of expanded bits, whose square is the bit difference.

The direct uchar Hamming backend returned incorrect zero distances on this
Windows OpenCV 5.0.0 contrib 5.0.0.93 build. Hamming therefore always expands
packed bytes into zero/one float bits and indexes them using native Euclidean
distance. This preserves Hamming rank and supports full 256-bit descriptors;
the direct Hamming binding remains unqualified. Expansion has explicit element
and dimension budgets. See [native references and fixture sources](<../examples/opencv tools/native_embedding_index/SOURCES.md>).

Enable result filtering to apply an inclusive Maximum distance or Minimum dot
similarity. Minimum results governs each query's pass/fail result. Rejected
queries retain neighbor diagnostics, emit null IDs/labels in the row lists and
withhold scalar Item ID/Label events. Scalar connectors also remain absent for
multiple queries. Distance/similarity and neighbor availability are not calibrated
confidence or a final ticket judgment.

## Native build, persistence and lifecycle

Choose a fixed tree count (1..64), uint32 seed, normalization and Memory/Disk
build. The disk mode calls `setOnDiskBuild` on a private temporary file before
adding rows; the memory mode calls native `save`. Both close and reload the native
file to verify item/tree counts before emitting a model. Small galleries can
report additional roots after save/load; Report retains the requested count and
observed reloaded count instead of assuming they are equal. Automatic tree count
is omitted so the pre-build resource estimate uses a fixed bound.

A self-contained JSON bundle stores exact float32 descriptors, IDs, labels,
contracts, settings, ABI information and base64 native index bytes, with separate
native-byte SHA-256 and whole-model revision. Save publishes a complete new file
without replacing an existing one. Load rejects duplicate/nonfinite JSON, changed
checksums, inconsistent metadata and incompatible ABI before native use, then
verifies a real native mmap load. Save As copies this one sealed model asset.
Image sources follow the application's ordinary path-rebasing rules.

Native binary formats are build/platform dependent. The bundle requires the same
OpenCV version, OS, architecture and byte order. Rebuild from the source gallery
on another platform/build; retained rows and metadata make that source data
inspectable. These are trusted native model files: checksums detect modification,
not authorship, and native parsing is not isolated in a separate process.

Search caches a private verified file and native mmap by revision/prefault.
Reset/close/native errors release the mmap before removing its private directory.
Native handles never travel through graph connections, persist in project JSON,
or transfer between worker copies. Prefault is a request; this platform may not
support it. Native build/query methods may use internal parallelism; this API does
not expose their build worker count and does not change global OpenCV settings.

Limits include 4,096 gallery/query rows, 65,536 native dimensions, two million
native feature values, values within ±1e6, a conservative node-byte estimate,
32 MiB maximum native file, 64 MiB maximum JSON bundle and 64 million query
candidate/feature comparisons and 64 million requested native search visits. A limit violation emits no partial current result.
Disk build still receives the full bounded input matrix; it is not streaming
out-of-core descriptor acquisition. Large-gallery benchmarking, incremental
updates and cross-build binary compatibility remain future work.

## Examples

[Ten original projects](<../examples/opencv tools/native_embedding_index/README.md>)
reuse openly published synthetic stripe images with independent held-out
queries. HOG and 256-bit block-mean hashes exercise all five metrics, L2
normalization, RAM/disk training, stable filenames, unknown rejection and a wrong
preprocessing contract. Independent scalar/image-derived references verify
rankings and measures; native file reloads, relocation, XML, failed-input gating,
Debug/Lean and a cold Run All worker are checked. These controlled galleries
qualify API/dataflow behavior, not general part recognition or search speed.
