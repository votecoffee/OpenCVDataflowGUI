"""Shared editor colors. Semantic display types do not alter runtime compatibility."""

PORT_COLORS = {
    "image": "#e6b94f",
    "region": "#ba8aeb",
    "pose": "#66a9f2",
    "pyramid": "#dbce79",
    "depth": "#4cc6c6",
    "number": "#9ed27a",
    "data": "#a8b4c7",
    "any": "#c2cbd5",
}

PORT_LABELS = {
    "image": "Image · gold",
    "region": "Region / mask · purple",
    "pose": "Pose / transform / coordinate frame · blue",
    "pyramid": "Image pyramid · pale gold",
    "depth": "Depth image · turquoise",
    "number": "Number · light green",
    "data": "Other structured data / tickets · slate",
    "any": "Any / unresolved type · gray",
}

BLOCK_COLORS = {
    "Input": "#45c6ba",
    "Output": "#ed976f",
    "Image processing": PORT_COLORS["image"],
    "Regions": PORT_COLORS["region"],
    "Poses": PORT_COLORS["pose"],
    "Inspection": "#c49bba",
    "Calibration": "#6ac99a",
    "Control flow": "#8fa8bd",
    "Sequence": "#869fea",
    "Data": PORT_COLORS["data"],
    "Recipes": "#d391ca",
    "Advanced": "#c29571",
    "ToolBlocks": "#9399ed",
    "ML": "#e18cc8",
    "Custom": "#c2cbd5",
}


def port_type(block, port, output=False, seen=None):
    """Resolve renamed ToolBlock terminals to their original semantic display type."""
    if port.visual_type in PORT_COLORS:
        return port.visual_type
    seen = set() if seen is None else seen
    key = (id(block), port.name, output)
    if key in seen:
        return "any"
    seen.add(key)
    bindings = getattr(block, "output_bindings" if output else "input_bindings", {})
    binding = bindings.get(port.name)
    if binding is not None:
        child = block.graph.blocks[binding.block]
        original = next(
            p for p in (child.outputs if output else child.inputs) if p.name == binding.port
        )
        return port_type(child, original, output, seen)
    name = port.name.casefold()
    if port.kind == "image" and (name == "mask" or name.endswith("_mask")):
        return "region"
    if port.kind == "data":
        if name == "region" or name.startswith("region_") or name.endswith("_region"):
            return "region"
        if name in {"pose", "transform", "frame", "coordinate_frame"}:
            return "pose"
    return port.kind if port.kind in PORT_COLORS else "any"


def block_type(block):
    if block.subtype in ("Regions", "Poses"):
        return block.subtype
    category = block.block_type or block.category
    return category if category in BLOCK_COLORS else "Custom"


def connection_type(source, target):
    """Concrete output types take precedence; generic outputs use the target hint."""
    return target if source == "any" else source
