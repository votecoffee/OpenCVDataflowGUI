# Visual-word image retrieval

Open a project and choose **Run all**. Included codebooks and galleries work offline
without plugins or downloads. Images and models are original MIT controls; see
[sources](SOURCES.md) and [the contract](../../../docs/visual-words.md).

| Project | Demonstration |
| --- | --- |
| [train_sift](train_sift.json) | Six images → SIFT → Batch → native k-means codebook → Save |
| [train_orb](train_orb.json) | Six images → ORB → Batch → binary representative words → Save |
| [gallery_sift](gallery_sift.json) | Encode three reference views, label and save a reusable gallery |
| [gallery_orb](gallery_orb.json) | Build the matching ORB/Hamming gallery |
| [retrieve_sift](retrieve_sift.json) | Rank three held-out views against the saved gallery |
| [retrieve_orb](retrieve_orb.json) | Same queries with binary descriptors and visual words |
| [hellinger](hellinger.json) | Reuse a gallery of square-root frequency signatures |
| [strict_rejection](strict_rejection.json) | An intentionally strict 0.001 cosine limit rejects a known view |
| [blank_query](blank_query.json) | No features gives no ranking and a valid inspection failure |
| [wrong_features](wrong_features.json) | ORB features with a SIFT codebook produce a contained error |
| [wrong_histogram](wrong_histogram.json) | L2 signatures with an L1 gallery produce a contained error |
| [spatial_rearrangement](spatial_rearrangement.json) | Shuffled quadrants still retrieve washers: spatial layout is not verified |

Training saves `trained-codebook-*.json` after six inputs. Gallery building saves
`trained-gallery-*.json` after three. Existing destinations are preserved; choose
new output paths before repeating. To use a retrained vocabulary, select its
saved codebook both when building the new gallery and when retrieving queries,
then select that newly saved gallery. A gallery from another codebook is rejected.

Each image retains at most 200 local features; the vocabulary has 16 words.
Six training views use ±3° rotation and separate noise. Held-out queries use 7°,
different noise and gain 0.95. All three intended labels rank first for SIFT,
ORB and square-root signatures. Best raw-frequency cosine distances are
**0.0119/0.0321/0.0396** (SIFT) and **0.0057/0.0158/0.0871** (ORB).
These are controlled examples, not a production accuracy or throughput result.

SIFT words use native float32 k-means and Euclidean matching. ORB words are actual
sampled byte descriptors selected by farthest-first Hamming distance. They are
not averages of descriptor bytes. Histograms discard spatial order: the rearranged
query still ranks as washers. Use geometric verification for pose or layout checks.

The current image stays unchanged; the retrieval label is a drawable overlay for
the filmstrip. Debug retains plots/graphics and Lean preserves numerical decisions.
`benchmark.py` independently checks native descriptors, NumPy assignments and
rankings; results are recorded in [benchmark.json](benchmark.json).
`build_example.py` reproduces images/models/projects and preserves an existing
assets folder.
