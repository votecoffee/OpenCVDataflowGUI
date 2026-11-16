# Image codec preflight

**Probe image file and codecs** (`cv.probe_image_file`) is under
**OpenCV → Image Codecs → Codecs**. It checks native file-reader detection and
page/frame inventory, an encoder extension, or both. Run it with **Run once**;
it does not acquire an image or create a source ticket. A wired **Path** overrides
the configured filename. Existing tickets and arbitrary connection payloads are
retained on the metadata results.

## What the results establish

| Output | Meaning |
| --- | --- |
| **Reader available** | Native signature check recognized an available file reader. |
| **Page count** | Native `imcount` inventory; zero means recognized headers could not be counted. |
| **Writer available** | The installed encoder registry accepts the chosen extension. |
| **OK** | Every requested check and page-count policy passed. |
| **Info** | Snapshot identity, native flags/count, encoder extension and structured diagnostics. |

Unrequested or unperformed optional checks emit no value. For example, a file
over the byte limit has no reader/count observation, and Writer mode has no file
observations. A recognized but truncated PNG can report Reader available true
and Page count zero. Missing/unsupported files, failed native probes, unavailable
encoders and page-policy failures produce valid failed inspections with reasons.
Invalid configuration produces the normal validation error without output.

**These are preflight checks, not pixel validation.** A corrupt compressed pixel
stream can pass header inventory. An available encoder does not establish support
for a particular depth, channel layout, compression option, output directory or
permission. Info always reports `pixel_decode_verified=false`; actual readers
and writers still own those checks and errors.

## Snapshot, counts and controls

The block reads a regular file through Python, bounds its size before native
calls and checks for changes during the read. It hashes the complete snapshot
and probes a temporary copy of those same bytes. The installed Windows native
file APIs cannot directly open some Unicode names; staging also avoids that
source-name problem. Temporary copies are removed after successful or failed
checks. Original files are never written by this tool.

Info records the OpenCV version, source path/extension, snapshot byte count and SHA-256, signature
prefix, requested count flag and native inventory. Reader selection follows the
signature, not the filename extension. A source may change after the snapshot;
the probe does not lock it or bind a later independent reader to these bytes.

**Mode** chooses File, Writer or File and writer. File checks use the configured
project/working-directory path base. Save As retains the ordinary input reference.
Writer checks accept a common-format dropdown or one **Custom extension**,
including formats supplied by another native build. They do not create a
destination image. Optional formats such as OpenEXR, AVIF or JPEG XL can be absent.

**Count flags** passes one of Unchanged, Any color, Color, Grayscale or GDAL to
native `imcount`. It does not convert image pixels. GDAL availability, selected
driver and count behavior depend on the native build; selecting that flag does
not qualify a GDAL backend. Native page/frame inventory is qualified here for
ordinary stills and TIFF/APNG/GIF/WebP sequences on the installed OpenCV 5 build.

**Maximum file bytes** bounds the snapshot up to 128 MiB. **Minimum pages** and
**Maximum pages** are inclusive policy limits applied to the returned native
count. The page limit does not cap native directory traversal, codec-internal
metadata allocations or execution time. This tool does not isolate native codec
processing in a subprocess. Live last-input-image preview is disabled because
execution reads the current file; normal graph runs perform a new snapshot.

[Five original examples](<../examples/opencv tools/codec_probe/README.md>) ·
[OpenCV reader/writer/count APIs](https://docs.opencv.org/5.0/main_modules/imgcodecs.html) ·
[Multipage memory transport](multipage-memory.md) · [Image formats](image-formats.md)
