# Group detections by spatial votes

Connect a detection collection to **Group detections by spatial votes** after
decoding or mapping all candidates to the same source image. An optional image
adds overlays and passes through with its pixels intact. A collection can run
without an image if only the representative boxes and vote counts are needed.

Boxes are processed separately by class ID. The block uses the visible XYWH
box and rejects rotated or incompatible image contexts. Choose **Cover** to
round pixel edges outward or **Nearest** to round each edge to the nearest
integer. Collapsed integer boxes are invalid. Output rectangles use integer
source pixel edges. The report retains the exact native rectangle. A rounded
representative at the border is clipped for the visible output.

OpenCV groups rectangles using a relative tolerance on all four edges and
returns the rounded average rectangle and the number of votes. The block
reconstructs the similarity components and verifies their average and vote
count against the native result before assigning source IDs. Each current
input box contributes one vote. The result preserves the member IDs in a
stable order, uses the first stable member as the group ID, and emits vote
count as both a dedicated field and the sortable Strength metric.
[OpenCV rectangle grouping](https://docs.opencv.org/4.x/de/de1/group__objdetect__common.html)

The similarity tolerance uses epsilon times half the sum of the smaller
width and smaller height of the two boxes. A connected chain can link boxes
whose endpoints are farther apart. Use a tolerance appropriate for the
closest parts. Zero tolerance groups exact duplicates of the quantized boxes.

**Minimum votes** is the smallest accepted multi-vote component. OpenCV can
also suppress a small component contained in a stronger same-class component.
The report separates that outcome from low vote counts. **Singleton policy**
can retain isolated single votes independently of the multi-vote threshold.
It does not retain a component of two votes when three are required.

The representative score is the maximum member score when all scores are
available and use the same declared score kind. The score is omitted if its
members cannot be compared. A retained singleton preserves its original score.
Vote counts measure repeated box proposals.
The count gates provide a tool verdict and do not assign a final ticket
judgment by themselves.

The [example projects](<../examples/opencv tools/spatial_vote_grouping/README.md>)
demonstrate adjacent parts, overlapping classes, minimum votes, and image-free
grouping.
