# VegQual attribution and changes

The image crops and publisher labels come from **VegQual: A Multiclass Dataset
for Real-Time Fresh and Defective Vegetables**, version 4, by Tarek Rahman,
Md Rahadul Islam Jishan, Robiul Islam, Md Rakibul Islam, Jannatul Tajrian,
Md. Sakhawat Hossain, Suman Ahmmed and Ohidujjaman.
[Figshare dataset](https://doi.org/10.6084/m9.figshare.30596084.v4) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [publisher API](https://api.figshare.com/v2/articles/30596084/versions/4)
was checked on 2026-11-18 and reports version 4 and CC BY 4.0.

This increment reuses the existing public crops; no archive download was needed.
Each source frame had one publisher-provided YOLO box. Existing preparation
cropped that box, resized the long edge to 192 pixels and JPEG encoded the crop.
`vegqual/ground_truth.json` preserves original image/annotation paths, normalized
box, label and publisher split. Training/validation/test divisions and the
selection of one source filename stem before the `.rf.` suffix are unchanged.
The available metadata does not identify physical specimens.

Additional modifications are ranked-score review PNGs and the diagnostic gallery.
An unchanged crop is copied to `duplicate.jpg` and deliberately assigned to a
conflicting validation class during the checker, then removed from assignments.
`unreadable.png` is authored ASCII fixture data, not a photograph. Neither fixture
is used to fit the final model. Image derivatives retain CC BY 4.0. No endorsement
is implied. [Original preparation notes](vegqual/SOURCES.md) include archive
identity and the publisher's ground-truth sources.

Authored code, the diagnostic fixture and the bundled tool use the included
[MIT license](LICENSE). The wheel includes its own MIT notice. No private
implementation-repository attribution is included in these public examples.
