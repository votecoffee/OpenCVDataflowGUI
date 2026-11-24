# Motion history and direction

Open [motion_history.json](motion_history.json) and choose **Run All**. This
original MIT clip shows a bright part moving right, then down. The threshold
block makes the part silhouette. The filmstrip shows the fading motion trail.
Select the history block to inspect age, direction, or labeled regions.

| Check on the example | Result |
| --- | --- |
| Rightward motion at frame 12, 600 ms | 0° and a 432-pixel region |
| Exactly 250 ms after rightward motion ends, frame 17 | 240 pixels retain history |
| Following frame at 900 ms | History cleared |
| Downward motion at frame 32, 1600 ms | 90° and a 527-pixel region |

**Motion history and direction** is under **OpenCV → Motion Templates**.
Give it a uint8 silhouette with nonzero pixels where motion occurs. Connect a
capture timestamp in milliseconds or use time from the video frame payload.
Each stream keeps its own history. Reset, a backwards timestamp, or a change
of clock source starts the history again.

**Recency** starts at 1 for newly observed motion and fades to 0 at the time
limit. **Age** reports elapsed milliseconds and uses −1 for inactive pixels.
**Active** marks pixels with history, including pixels exactly at the time
limit. Masks and regions limit updates and visible history. The block keeps
capture times in float64 and uses a short relative clock for OpenCV's float32
motion templates to preserve precision for large timestamps.

**Orientation mask** marks valid local directions. The dominant direction
is clockwise from image-right. Region labels grow from the current
silhouette within a neighbor time tolerance. The report lists the largest
regions and the label image includes every region.
[OpenCV motion templates reference](https://docs.opencv.org/5.0/extra_modules/motempl_8hpp.html)

Rebuild the clip and project from the repository root.

    python "examples/opencv tools/motion_history/build_example.py" --output "examples/opencv tools/motion_history"
