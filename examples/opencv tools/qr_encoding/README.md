# Native QR label generation

Open a project, then select **Run**. The ordinary and Unicode examples generate
a label and read it back through **Read QR codes**.

- `qr_roundtrip.json`: uppercase part/lot text with H correction; its pristine
  image matches `label_reference.png` exactly.
- `qr_unicode.json`: a UTF-8 ECI payload with accented and Japanese characters.
- `structured_append.json`: generate three native structured-append symbols,
  delivered together as a `FlowGroup` and shown in a contact sheet. This project
  demonstrates generation; ordered recovery remains roadmap item S1-162.
- `capacity_failure.json`: intentionally requests too much byte data for version
  1. It reports invalid configuration and emits no image. Its CLI exits with failure.

Select the generator or reader to inspect images and data. These projects create
no output files. Add **Save image** with a new destination to export the pristine
`image`; result-graphics bounds are separate overlays.

## Controls and outputs

**Generate QR labels** is under **OpenCV → Object Detection (objdetect) → Code
generation**. A wired `text` input overrides the configured text and carries its
ticket and companion data. Label images have new coordinate grids; camera,
pyramid and raw-sensor coordinates are not reused for generated pixels.

All native parameters are exposed: automatic or explicit version 1–40, L/M/Q/H
correction, encoding mode and structured-append count 1–16. Choices use dropdowns.
Numeric and alphanumeric modes validate their character sets. Byte/ECI inputs
use UTF-8. Kanji mode converts validated double-byte Shift-JIS; the native Python
reader cannot return its text, so use a compatible reader such as the optional
ZXing adapter. This build's ECI reader emits a native warning while recovering
the tested UTF-8 payload. The API does not expose arbitrary segment lists or an
ECI assignment selector. See the [OpenCV encoder API](https://docs.opencv.org/5.0/main_modules/classcv_1_1QRCodeEncoder.html).

Integer **Pixels per module** scaling preserves binary edges. **Quiet zone** is
the total blank border in modules, replacing the native border; default four.
Zero is allowed for deliberate composition. Sheet columns and gap arrange
multiple labels. Combined symbol/sheet pixel and axis limits are checked before
scaling; runtime and GUI snapshots add memory beyond that allocation budget.

- `image`: one label, or the contact sheet for multiple symbols.
- `symbols`: the actual individual images in a `FlowGroup`, in native output
  order, with separate frames and group/position metadata.
- `modules`: an optional unscaled, border-free matrix for one symbol. It is
  withheld when multiple symbols are generated.
- `count`, `text`, `info`: symbol count, original payload and rendering details.

Structured-append generation currently accepts ASCII and reports
`requires_reassembly=true`. A part's reported position is its index in the native
output vector, not an independently decoded header. Individual readers may
return fragments. Neither those fragments nor the contact sheet establish a
complete recovered payload; missing/duplicate/mismatched fragment handling is
reserved for S1-162. Generation pass/fail is not a print-quality grade or a
scanner/lighting robustness qualification.

## Fixture and validation

All payloads, graphs and the reference image are original MIT-licensed fixtures,
created by `build_example.py` using native OpenCV. No customer or third-party
images are included. Every graph contains an attribution/scope comment.

`truth.json` records exact input text independently of the readers. Targeted
checks cover native modes/correction/version parity, exact binary rendering,
Unicode recovery, ticket/grid ownership, grouped sheet geometry, bounded
allocations, capacity/validation failures, graphics-off and XML presets. All
four projects run in Debug and Lean with their expected pass/failure outcomes.
