# Canvas colors

Connections and sockets share a display-type palette:

| Data | Color |
| --- | --- |
| Images | Gold |
| Regions and binary masks | Purple |
| Poses, transforms and coordinate frames | Blue |
| Image pyramids | Pale gold |
| Depth images | Turquoise |
| Numbers | Light green |
| Other structured data, including tickets | Slate |
| Generic `any` / unresolved | Gray |

Wires use the source output's display type. A generic `any` output uses the input's
display type when available. Generic flow blocks can carry different kinds across
runs, so unresolved generic ports stay gray. Colors do not change dynamically with
the last value. A pyramid feeding an image input keeps its pale-gold wire, showing
that an image level is being selected. A binary mask is still an image at runtime
but has a purple display hint. A mask wired into an ordinary image input therefore
has a purple wire and a gold destination socket.

Structured pose/region data still uses the `data` runtime contract. Semantic names
such as `pose`, `transform`, `region`, `region_a`, `training_region`, `mask` and
`search_mask` supply display hints. Colors do not add new runtime type checking.
Renamed public ToolBlock terminals resolve their original internal terminal's
display type, including nested ToolBlocks. Custom Python tools can declare an
explicit hint for arbitrary names:

```python
Port("inspection_origin", "data", visual_type="pose")
Port("search_window", "data", required=False, visual_type="region")
```

Supported hints are the keys in `visual_style.PORT_COLORS`; unrecognized hints
fall back to the ordinary kind/name rules. Hints are code metadata and do not
require saved-project migration. Custom plugin code supplies its metadata when
loaded, just as it supplies the port's ordinary kind.

Block borders indicate the toolbox category, with specific purple **Regions** and
blue **Poses** subtypes. Input tools use teal, outputs coral, image processing gold,
inspection mauve, calibration mint, control flow blue-gray, sequence memory indigo,
data slate, recipes pink, ML rose, advanced tools tan and reusable ToolBlocks violet-blue.
Unrecognized custom categories use neutral gray. The minimap uses these same block
category colors.

Node **headers** retain run-status colors: idle slate, pass green, fail orange,
error red, skipped muted purple, waiting blue and inactive muted blue-gray. Scores
adjust header intensity. Selecting a node or wire adds a white outline while
preserving its type color. Temporary connection drags use the originating socket's
color and a dashed line. Wire widths remain legible when zooming out.

Open **Workspace > Color legend…** for an in-app reference. Names, port tooltips,
run-state text and border/selection geometry accompany colors; colors are not the
only indication of type or status. This palette replaces the previous cyan image
sockets and uniform teal wires. Filmstrip borders still indicate ticket verdicts.
