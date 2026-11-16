# Structured-append QR recovery

**Read structured-append QR** (`inspect.read_qr_append`) recovers one complete
multipart payload from a scene or an atomic image group. It uses OpenCV's
standard/ArUco detection and per-symbol byte decoding. It reads the actual QR
headers; generator metadata, filenames and visual/input order do not supply
fragment identity. No optional decoder dependency is required.

Find it under **OpenCV → Object Detection (objdetect) → Code reading**. The
[eight examples](<../examples/opencv tools/qr_structured_append/README.md>) include
shuffled success cases and deliberate incomplete/ambiguous failures.

## Inputs and settings

Connect exactly one of **Image** or **Images**. Image accepts one uint8 gray or
supported tagged color scene/contact sheet. Optional mask/region inputs intersect
in the source grid and whiten unselected pixels; keep each symbol and its quiet
zone inside the selection. Images accepts one `FlowGroup` of up to 16 images.
Use Batch serial entries upstream to accumulate separate acquisitions, then run
the reader once on that complete group. Apply selections per member upstream.
The reader stores no partial set between invocations and performs no acquisition.

| Setting | Behavior |
| --- | --- |
| Backend | ArUco by default; standard is also available. Both expose the existing [native finder controls](qr-backends.md). No fallback. |
| Mode | Multi detection only. Each candidate is individually decoded so native multi-code concatenation cannot conceal missing/duplicate fragments. |
| Expected parts | Zero accepts any complete 2–16-part set; a configured count must match every header. One-part sequences are rejected. |
| Expected SHA256 | Optional exact 64-hex-digit digest of the full payload bytes. Use it when the intended payload is known. |
| Text encoding | Strict ASCII, UTF-8 or hexadecimal after successful byte assembly. No replacement decoding in the final payload. |
| Maximum images | Atomic group entry cap, default 16. |
| Maximum pixels / Total pixels | Per-image and combined input caps, defaults 16,777,216 / 33,554,432. Checked before native detection. |
| Maximum symbols | Candidate cap, default 16; checked across all images before any per-symbol decoding. |
| Maximum payload bytes | Sum of decoded fragment bytes, including duplicates/rejected candidates; default 65,536. |
| Draw overlay | Single-image fragment outline, label and image-plane axes. Group results retain geometry separately for each member. |

The full [parameter reference](block-parameters.md#read-structured-append-qr-inspectread_qr_append)
includes ranges and conditional native controls. Invalid configuration/input or a
native exception produces no current outputs through the common block executor.
A later valid invocation can recover. These caps do not bound all native memory,
runtime/editor input copies or detector execution time. Search grayscale buffers
can be retained up to the combined pixel cap while the candidate count is checked.

## Verification and result contract

For each decoded binary module grid, the reader checks the two exact format
copies, removes the QR mask/function modules, deinterleaves Model 2 codewords and
checks Reed–Solomon syndromes. It then reads the leading structured-append mode,
zero-based index, declared count and eight-bit parity. Block-layout tables cover
versions 1–40 and correction levels L/M/Q/H.

This is a conservative header verifier, not another error-correcting decoder:
damaged format/codewords are rejected even when native OpenCV can repair the
payload. Ordinary QR codes without a leading multipart header are rejected.
The test set includes nine versions spanning 1–40 at all four correction levels
(108 native module grids), including short/long interleaved RS blocks.

Assembly requires one consistent count/parity, exactly one of every index, no
unverified or additional fragment, and matching payload XOR parity. Missing
indices, duplicate indices and incompatible counts/parities are reported. Raw
bytes are concatenated in header-index order, then the optional expected SHA256
and selected text encoding are checked.

**QR parity is only eight bits.** Different payloads can share it, including mixed
sets whose fragments also preserve XOR parity. Without an independently known
digest or application-level identity inside the payload, complete headers and
parity cannot distinguish every such mixture. The collision example demonstrates
that limitation and rejects it using the expected full-payload hash. A hash pin
identifies configured bytes; it is not a cryptographic signature or sender proof.

| Output | Contract |
| --- | --- |
| Text / Bytes hex | Emit only for a fully verified, correctly decoded set. Incomplete, duplicate, mismatched or encoding-failed sets emit neither. |
| Complete | Boolean judgment for this invocation. |
| Fragments | All candidates with individual bytes, actual header or validation error, original-image corners/center, member index/label, and each member's ticket/grid. |
| Status | Counts/parities, missing/duplicate indices, rejection reasons, digest-check setting, backend, image count and decoded-byte total. |
| Overlay | Single-image source grid/ticket with rendered fragment diagnostics; absent for image groups. |

Successful groups can contain different acquisition tickets. Their records retain
each member's ticket and grid; group labels are not a registration or exposure
synchronization claim. Do not interpret member coordinates as one camera image
or a calibrated 3D pose. Single-image drawable result graphics map to the original
image/filmstrip; failed aggregate graphics are amber and passing graphics green.
Disable Draw overlay and optional graphics to avoid their drawing work.

Inspection failures keep Status/Fragments/Complete outputs, with reason codes such
as `missing_qr_fragments`, `duplicate_qr_fragment_index`,
`mismatched_qr_fragment_set`, `unverified_qr_fragment` or
`qr_expected_digest_mismatch`. Downstream failed-input handling follows the shared
runtime policy. Configuration/algorithm exceptions are a separate error state.
The block uses the common timing, ticket reporting, logger and exception boundary.
Independent native detector instances support parallel branches and isolated previews;
JSON/XML store configuration without retaining an unfinished message.

## Sources

Native [GraphicalCodeDetector](https://docs.opencv.org/5.0/javadoc/org/opencv/objdetect/GraphicalCodeDetector.html)
provides per-symbol bytes and module grids. The installed OpenCV/ZXing Python
interfaces do not expose native sequence headers directly. The local verifier
fills that gap using module data rather than trusting input-side labels.

QR block-layout tables derive from Project Nayuki's MIT-licensed
[QR Code generator library](https://github.com/nayuki/QR-Code-generator/blob/bcfe1b018cec5f1f91b5705eca5bb4cdd34e9f8d/python/qrcodegen.py).
Only its two numerical tables are incorporated; the full upstream copyright and
permission notice is retained in the packaged `qr_rs_tables.py` module. Header
traversal, validation and block logic are local implementations. All example
images/payloads/projects are original MIT work with on-canvas attribution.
