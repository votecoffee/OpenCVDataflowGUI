# Constrained OCR controls

Open a project in this folder, then Run once. The source image and per-read
graphics feed the filmstrip. These focused tool examples use original synthetic
images with known truth; they do not measure industrial OCR accuracy.

| Project | Expected result |
| --- | --- |
| `beam_recovers.json` | Pass: prefix search recovers A although each step favors blank. Exact A mass is 0.688. |
| `greedy_blank.json` | Fail: the same image's raw greedy read is blank. |
| `repeat_blank.json` | Pass: A/blank/A preserves two labels, AA. |
| `blank_dominant.json` | Fail: the lexicon permits A, but blank has more model mass. |
| `unreachable_word.json` | Fail: AAAA cannot align to three time steps. |
| `confidence_limit.json` | Fail: A's 0.688 confidence is below 0.9. |
| `lexicon_file.json` | Pass: the pinned UTF-8 word file allows A and AA. Save As carries it. |
| `alphabet_check.json` | Fail: raw greedy ABA violates alphabet A; it is not silently shortened. |
| `lexicon_check.json` | Fail: raw greedy ABA is outside lexicon BA; it is not replaced. |
| `ratio_guard.json` | Fail: constrained AA has insufficient mass relative to raw greedy ABA. |
| `native_lexicon.json` | After preparation, pass: unchanged CRNN reads lot2048, pnax12 and pass. |

The first ten projects need no model download. Their tiny native ONNX models
perform image-dependent arithmetic, not learned text recognition. `truth.json`
independently enumerates every alignment for the three-step probability controls
and retains image/model hashes. Each graph contains an attribution comment.

For the optional native glyph example, from the repository root run:

```powershell
.venv/Scripts/python.exe "examples/opencv tools/constrained_ocr/prepare_model.py"
```

This explicitly downloads one unchanged Apache-2.0 OpenCV Zoo CRNN into ignored
`.local/crnn_en`, checks its exact bytes/hash and seals its preprocessing,
constraints and execution settings. To reuse identical local weights, append
`--model path/to/model.onnx`. An existing different bundle is preserved and
rejected. The export is fixed N=1: three labels execute three ordered calls,
without hidden batch fallback. No weights or customer images enter this repo.

Configure Vocabulary, Allowed characters, inline Lexicon or a pinned Lexicon
file, Beam width, optional Token top K and score/rejection controls in the block
editor. Exact sequence mass is not calibrated correctness; finite search can
miss candidates. See the [full contract](../../../docs/constrained-ocr.md) for
score semantics, resource guards, alternative reads and source metadata.

Developer checks: `tests/test_ctc_search.py`, `tests/test_ocr_constraints.py` and
`tests/test_constrained_ocr_examples.py`. Regenerate with `build_example.py` in
the development environment (requires optional `onnx`); differing existing assets
are preserved. See [SOURCES.md](SOURCES.md) and the retained upstream license.
