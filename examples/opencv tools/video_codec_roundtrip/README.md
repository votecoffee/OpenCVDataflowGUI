# Video codec round trip

Open `project.json` and **Run all** to record eight original RGB-tagged PNGs
as BGR FFV1 AVI at **12.5 FPS**. Execution closes the recorder and finalizes
`recorded.avi`. Open `replay.json` and **Run all** to review the decoded frames.
The blue, green and red bars must retain their colors. The white part moves
nine pixels per frame. Frame tickets identify `recorded.avi#000000` through
`#000007`; nominal media times are 0, 80, ... 560 ms. FFV1 decoding must match
the PNG pixels exactly.

Existing recordings are preserved. Move your previous recording or choose a
new destination before recording again, and update the replay path if needed.
The generated AVI and local qualification directory are ignored by Git.

From the repository root, run the six-case native qualification matrix with:

```powershell
python -m opencv_dataflow.video_qualification --output-dir "examples/opencv tools/video_codec_roundtrip/qualification"
```

The destination must be a new directory. It receives the clips and `report.json`.
The report checks finalized frame count, EOF, order, rate, media time, channel
errors and independent moving-part centers. FFV1 requires exact pixels;
MJPG and mp4v have explicit lossy tolerances. Missing codecs are reported as
`unavailable`, make the command fail, and remain unqualified. Use `--case`
to select individual combinations on another OpenCV build.

The qualified matrix is FFmpeg FFV1 and MJPG in BGR/grayscale at 12.5 FPS,
FFmpeg mp4v BGR MP4 at 12.5 FPS, and OpenCV MJPEG BGR AVI at 10 FPS.
The supported build's OpenCV MJPEG backend rounded fractional FPS and emitted
corrupted grayscale streams during native probing. The recorder rejects
these combinations, including when automatic backend selection chooses it.
Select FFmpeg for grayscale or fractional FPS.

Images, generator and projects are original content under the repository's
MIT license. `assets/truth.json` records analytic geometry and color samples.
Regenerate fixtures with `python "examples/opencv tools/video_codec_roundtrip/generate.py"`.
The native matrix uses generated original frames and requires no downloaded assets.

See [video workflow documentation](../../../docs/video-files.md) and
the [OpenCV VideoWriter reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1VideoWriter.html).
