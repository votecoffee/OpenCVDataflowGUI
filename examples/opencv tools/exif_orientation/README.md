# Image metadata and EXIF orientation

Open [exif_orientation.json](exif_orientation.json) with **File → Open…**,
then click **Run once**. Two **Image file** blocks read the same synthetic
JPEG with EXIF orientation 6. One outputs the stored 80 × 48 pixel grid.
The other turns it clockwise and outputs a 48 × 80 image.

**Metadata mode** is under **Input → Files → Image file**.

- **Off** is the default and uses the ordinary image decode path.
- **Read** adds metadata to **Info** and keeps the decoded pixel grid.
- **Apply orientation** also applies EXIF rotations or reflections when
  OpenCV reports a supported orientation for a JPEG or PNG. The output
  coordinate frame uses the oriented dimensions.

The **Info** summary lists the types and sizes of chunks that OpenCV
reported, including EXIF, XMP, ICC, or CICP when available. It also reports
recognized EXIF fields such as orientation, make, model, software, and
capture date/time. Capture date/time is the stored EXIF text. Raw chunks
are omitted from the summary.
[OpenCV image metadata reference](https://docs.opencv.org/5.0/main_modules/imgcodecs.html#imdecodewithmetadata)

```powershell
opencv-dataflow-run "examples/opencv tools/exif_orientation/exif_orientation.json"
```

[All examples](../../README.md)
