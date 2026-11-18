# Sources and reproducibility

All images, models and projects here are original repository controls under the
root [MIT license](../../../LICENSE). They include no customer captures, external
weights or third-party image assets. Each graph includes an on-canvas credit note.

`build_example.py` draws washer-like rings, crosses and tabs on 256×256 monochrome
images. Each motif has two training views at -3°/+3° with noise seeds 10–15.
Held-out queries use 7° rotation, noise seeds 50–52 and gain 0.95. Query pixels
and hashes differ from every training image. Codebooks use only the six training
views; galleries use one training view per motif. Query images never contribute
to fitting or gallery construction.

The rearranged control permutes four quadrants of the held-out washer query,
preserving its pixel population while changing layout. Its retained first rank
demonstrates that local-feature histograms do not verify spatial arrangement.
The blank control and deliberately overstrict threshold expose empty/rejected
query behavior. Three simple known motifs do not establish field accuracy.

[truth.json](truth.json) records construction roles, expected labels and exact
asset SHA-256 values. Frozen codebooks/galleries retain content revisions.
[benchmark.py](benchmark.py) imports no application implementation: it independently
uses native SIFT/ORB descriptors, NumPy Euclidean/bit-count word assignment and
cosine ranking. Recorded [results](benchmark.json) include all three normal queries
and the rearranged query for SIFT, ORB and Hellinger signatures. ORB WTA 3/4
Hamming2 and alternative ranking metrics have separate focused numerical tests.
