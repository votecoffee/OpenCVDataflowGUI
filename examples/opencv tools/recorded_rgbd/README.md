# Recorded RGB-D

Open [recorded_rgbd.json](recorded_rgbd.json) and choose **Run All**.
The source pairs color and depth within 5 ms. Both streams contain four
frames, but only three pairs meet the tolerance. The source emits one ticket
per pair and reports both original acquisition timestamps. The filmstrip shows
the selected color frame for each ticket.

| Pair | Depth minus color time | Registered depth pixels | Confidence rejections |
| --- | ---: | ---: | ---: |
| 1 | +2 ms | 1,334 | 72 |
| 2 | +3 ms | 1,348 | 72 |
| 3 | +1 ms | 1,348 | 72 |

The depth camera is 48 × 32 and the color camera is 80 × 48. Inspect the
registration block's **Depth**, **Confidence**, or **Valid Mask** result to
see the color-grid projection. The exported XYZ cloud has one colored
point per valid registered pixel. The two intrinsic models and the depth-to-color
pose are visible in the graph. The rig and the images are synthetic, so the
example can verify software behavior without a camera or a synchronization device.

## Use your own recording

Use a versioned [manifest](assets/recording.json) with one named shared clock, the
two optical frame names, meters per raw depth count, and two arrays of
strictly increasing integer nanosecond timestamps. The source consumes the
earliest pair within tolerance in FIFO order and uses each frame at most once.
Set Maximum Skew to zero to require identical timestamps. Acquisition
timestamps must already be in a common time domain.

- Put PNGs in the manifest folder or its subfolders and give each record a
  relative path. Use three-channel uint8 or uint16 color and single-channel
  uint16 depth. Zero depth represents a missing measurement.
- Optionally attach a same-grid uint8 or uint16 confidence PNG to each depth
  record. Confidence uses the full range of the stored dtype, so 255 is full
  confidence for uint8. An absent confidence produces no confidence event.
- Supply depth intrinsics for the undistorted depth grid, color intrinsics
  for the target grid, and a metric depth-to-color pose. The nearest
  projected surface wins at each target pixel. A rejected nearest sample
  leaves a hole.
- Relative manifest paths resolve from the saved project. The source reads
  one selected pair at a time. Reset flow state to reload an edited manifest.

The original synthetic frames are by the OpenCVDataflowGUI contributors (MIT).
To regenerate them and the saved graph, run
`python build_example.py --output <new-directory>`.
