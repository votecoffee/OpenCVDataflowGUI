# Constrained OCR decoding

`ml.recognize_constrained_text` adds alphabet filtering and optional exact-word
constraints to [ordered region OCR](batch-ocr.md). It retains the same source
regions, rectification, native batch inference, shape controls, source image,
tickets, recipes, result graphics and model receipts. Multiple regions belong
to one acquisition; this block does not collect unrelated source tickets.

Try the [eleven saved projects](<../examples/opencv tools/constrained_ocr/README.md>).
Ten run without downloads. The native glyph example explicitly prepares unchanged,
hash-pinned Apache-2.0 CRNN weights and a sealed configuration.

## Constraints and decoders

Vocabulary still maps native classes to unique tokens, excluding the explicit
blank class. Allowed characters is case sensitive; empty allows all tokens.
Every character in a multi-character token must be permitted. Inline Lexicon
contains unique exact words. Alternatively supply a project-relative UTF-8
word-line file and its SHA256. The bounded file is read and hashed once per
activation, declared for Save As, and used unchanged for all regions. Inline
words are sealed in model settings. Empty region selection reads neither model
nor lexicon. No case folding, spelling correction or word-frequency prior is
applied. Characters outside the model/alphabet are a configuration error;
an in-vocabulary word with no possible alignment yields a current failed read.

- **CTC-greedy** checks the raw greedy read against the constraints. It never
  deletes forbidden characters or silently substitutes a word.
- **CTC-prefix-beam-search** retains separate blank/nonblank ending masses for
  label prefixes. Adjacent equal classes merge; equal labels separated by blank
  remain distinct. A character trie restricts extensions to lexicon prefixes.
  Beam width bounds retained prefixes. Token top K limits allowed nonblank
  classes per step; zero keeps them all, and blank is always considered.

The recurrence follows standard CTC prefix decoding, described in
[Hannun's CTC explanation](https://distill.pub/2017/ctc/). This implementation
uses independent exact rescoring and bounded alignment evidence after search.

## Scores, ranking and rejection

Softmax uses stable float64 log probabilities. Probability inputs must be finite,
within [0,1], and sum to one within 1e-4 at each step. Whole rows are normalized
only to remove that rounding error, which is reported. Alphabet and lexicon
filtering never renormalize allowed probability mass.

Each surviving complete label sequence is rescored with an exact CTC forward
sum over all its alignments, including alignments omitted by search pruning.
Ranking uses this unconditioned mass, with deterministic class-ID tie breaks.
Finite beam/top-K search can discard the globally best sequence; exact rescoring
does not restore missing candidates. Reports therefore make no global optimum
guarantee and expose pruning, length limits and actual search work.

Confidence mode selects sequence probability `P(labels | model output)` or
token normalized confidence `exp(log(P) / max(1, label count))`. The latter
is the default. This changes the acceptance score, never candidate ranking.
Multi-character vocabularies can spell the same text with different label
sequences: these remain separate candidates, not a summed text posterior.
Neither score is calibrated correctness. Retained log probabilities remain
available even when exponentiation underflows to zero. The raw greedy text,
its exact label-sequence mass and its older emitted-run peak confidence are
reported separately.

Reject blank dominant is on by default: a word must exceed the all-blank
sequence's mass. Minimum greedy ratio optionally rejects a constrained result
whose mass is too small relative to the unconstrained greedy label sequence;
the reported ratio is capped at one, and zero disables this gate. Minimum
confidence applies afterward. These controls reduce forced dictionary reads
but require qualification with actual labeled inspection images.

Reads retain current text, alternatives, source region, selected class IDs,
scores and reason codes. Status can be `read`, `blank`, `low_confidence`,
`skipped`, `constraint_miss`, `alphabet_miss`, `lexicon_miss`, `length_limit`,
`blank_dominant` or `greedy_ratio`. Count/Text include accepted reads only.
Minimum reads and aggregate All/Minimum reads retain the base block's behavior.
Failures still emit current read/report/image data; configuration/native errors
use the executor's contained exception path with no new output event.

Character evidence follows the selected sequence's best Viterbi alignment:
token/class, half-open time-step span and peak probability within that run.
These are model time steps, not source-pixel character boxes or uncertainty
intervals. Optional source-quad/text graphics use the read's pass/fail color.
Connect Result graphics to Overlay result graphics before Filmstrip image.

## Resources and qualification

Budgets bound lexicon words/characters, the 1 MB file snapshot, decoded labels,
temporary prefixes, attempted prefix/class transitions and combined time/state
work for candidate scoring, greedy scoring and selected alignment. Exceeding
a work budget raises a contained configuration/runtime error. Reaching the
decoded-length limit suppresses extensions and records the count. Ordinary
crop, tensor, region and time-step guards remain in force. These limits do not
bound native DNN workspace or process RSS.

Independent exhaustive path enumeration checks exact CTC mass, repeats, nonzero
blank IDs, tokenization, extreme logits and impossible sequences. Native
image-dependent ONNX fixtures exercise recovery, rejection and source payloads;
unchanged CRNN weights read three printed labels under a constrained lexicon.
Debug/Lean, resource recovery, sealed settings, file digests and JSON/XML
relocation are covered. These controlled fixtures establish software contracts,
not OCR accuracy on natural industrial scenes or physical equipment acceptance.
