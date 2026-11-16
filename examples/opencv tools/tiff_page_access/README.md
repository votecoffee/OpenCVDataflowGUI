# Read TIFF pages on demand

Five self-contained projects use an original **257-page mixed-depth TIFF**.
Open a project and use **Run once**. This tool reads a requested page; it does
not advance through a replay sequence. No camera, plugin or download is needed.

| Project | Expected result |
| --- | --- |
| [random_access.json](random_access.json) | The wired Counter input selects page 256 with Step=0. Repeat Run once for a cache hit and a fresh image ticket. Change the counter's Start to 3, 128, 256, etc. to request pages in arbitrary order without reconfiguring the TIFF block. |
| [converted_page.json](converted_page.json) | Read color page 129 as float32 RGB. Cache stores the original uint8 BGR samples. |
| [no_cache.json](no_cache.json) | Read page 256 with retention disabled; every activation decodes, with identical samples. |
| [page_outside_stack.json](page_outside_stack.json) | Page 257 is beyond the last page (256). Validation error and no output are expected. |
| [workspace_limit.json](workspace_limit.json) | Deliberately tiny decoded workspace budget rejects the page before pixel decoding. |

Inspect **Info → cache** for hits, raw resident bytes, LRU indexes, evictions and
snapshot read/load counts. Both page-count and byte limits apply. On change
checks file observations; Verify content hashes every request; Pinned keeps the
snapshot until reset or path change. A wired Reset=true forces a refresh.

```powershell
opencv-dataflow-run "examples/opencv tools/tiff_page_access/random_access.json" --repeat 2 --profile debug
opencv-dataflow-run "examples/opencv tools/tiff_page_access/page_outside_stack.json" --profile lean
```

The second command intentionally returns failure. Applying configuration to the
TIFF block may reset its cache; editing the separate counter leaves it available.
The fixed request list and raw pixel hashes are in `truth.json`. Tests verify
two hits/nine native page decodes for that list, with peak raw cache residency
196,608 bytes, rather than retaining the 13,393,920-byte raw stack. TIFF directory
traversal and runtime copies still cost work; no general speedup is claimed.
See [controls and limits](../../../docs/tiff-page-access.md).

All pages, formulas and project content are original OpenCVDataflowGUI contributor
work under the repository MIT license. Every project includes an attribution
comment block. There are no customer or third-party images. `build_example.py`
regenerates the TIFF, its SHA-256 manifest, all 257 raw reference hashes and the
five saved graphs deterministically on the supported build.
