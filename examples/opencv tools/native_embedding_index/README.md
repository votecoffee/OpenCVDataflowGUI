# Native image embedding index

Open one of these projects and press **Run all**. They use OpenCV's native index
without the optional Annoy Python package. Gallery and held-out images are
original public synthetic stripe controls.

| Project | Demonstrates |
| --- | --- |
| `euclidean.json` | Exact L2 reranking of native candidates and stable filenames. |
| `manhattan.json` | Sum-of-absolute-difference ranking. |
| `angular.json` | Angular descriptor similarity with nonzero-vector checks. |
| `dot.json` | Descending inner-product similarity. |
| `hamming.json` | 256-bit block-mean image hashes connected directly to native retrieval. |
| `normalized.json` | L2-normalized HOG gallery built using native on-disk mode. |
| `reject.json` | Diagonal query rejected by a zero-distance requirement; no recognized ID. |
| `wrong_contract.json` | Explicit error for mismatched HOG preprocessing. |
| `train_memory.json` | Six serial images batched into a named RAM-built gallery. |
| `train_disk.json` | Same named gallery using private native on-disk build. |

Select Search > Neighbors to inspect image IDs, class labels, exact distances or
dot similarities and raw native measures. Report includes missing candidates,
cache reuse and model revision. Result graphics put the nearest label on the
original image and filmstrip. Retrieval pass means enough neighbors met the
configured metric criteria; it is not independently verified class correctness.

Training projects emit a model after all six images. Connect **Save native image
embedding index** to Model and choose a new JSON path to retain it. The bundle
contains exact native bytes, descriptors, settings, hashes and ID mappings.
**Load native image embedding index** loads that single file in another graph.
Existing saved files are preserved. The examples' source/query images are distinct.

The shipped native bundles were built on Windows x64, OpenCV 5.0.0 contrib
5.0.0.93. A different native ABI is rejected; regenerate on that platform into
a new folder using `build(destination)` from `build_example.py`. The builder
preserves an existing assets directory and verifies the source image hashes.
The small supplied fixture regenerates exactly on the tested build; no general
byte-identical forest guarantee across machines is claimed.

The Hamming path uses native Euclidean on unpacked bits because the installed
direct byte-Hamming binding returned incorrect distances. Exact bit differences
are retained, including values above 255. See [contracts and limitations](../../../docs/native-embedding-index.md)
and [SOURCES.md](SOURCES.md). Synthetic orientations do not qualify general
recognition accuracy, industrial robustness or a speed benefit over exact search.
