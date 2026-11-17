# Sources and licenses

The JPEG crops, publisher labels and annotation-derived ground truth are adapted
from **VegQual: A Multiclass Dataset for Real-Time Fresh and Defective Vegetables**,
version 4, by Tarek Rahman, Md Rahadul Islam Jishan, Robiul Islam, Md Rakibul Islam,
Jannatul Tajrian, Md. Sakhawat Hossain, Suman Ahmmed and Ohidujjaman.
[Figshare v4](https://doi.org/10.6084/m9.figshare.30596084.v4) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [publisher API](https://api.figshare.com/v2/articles/30596084/versions/4)
license name and version were rechecked on 2026-11-17; they report CC BY 4.0 and v4.

The existing public example cropped the single publisher-provided YOLO box from
each selected frame, resized its long edge to 192 pixels and encoded JPEG.
`vegqual/ground_truth.json` records every original image/annotation member,
normalized box, publisher class label and split. Labels preserve the publisher's
case, including `defected_Potato`. `vegqual/SOURCES.md` retains the earlier
provenance and archive MD5. This example copies those crops and the two already
trained models unchanged. No new training, threshold adjustment or test-image
selection was used to improve these results.

The four selected classes each have 24 training, five validation and four test
crops. Publisher splits are preserved, original filename stems deduplicated and
identical crop pixels excluded. Metadata identifies source frames, not individual
physical specimens. Held-out scores are descriptive for this small subset.

Ranked assessment PNGs, frozen NPY/JSON evidence and trained OpenCV models are
derived from these images; the same attribution applies. Test labels are used
only for evaluation receipts, never as an input to runtime classification.
Code and bundled wheel retain their [MIT copyright/license](LICENSE).
