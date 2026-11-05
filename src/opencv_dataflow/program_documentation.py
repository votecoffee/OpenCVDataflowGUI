"""Export an offline HTML guide to the applied configuration of a graph."""

import json
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile

from .parallel_inspection import ParallelInspect
from .port_types import connection_mode
from .toolblocks import ToolBlock
from .visual_style import BLOCK_COLORS, block_type
from .workspace import arranged_positions

STYLES = """
:root { font: 16px/1.5 system-ui, sans-serif; color: #172635; background: #fff }
body { margin: 2rem auto; padding: 0 1rem; max-width: 75rem }
h1, h2, h3, h4 { line-height: 1.2 }
a { color: #175188 }
section.block { border-top: 1px solid #cbd5e1; padding: .5rem 0 1rem }
section.scope section.scope { border-left: 3px solid #cbd5e1; padding-left: 1rem }
.diagram { overflow: auto; border: 1px solid #cbd5e1; background: #f7fafc }
.diagram svg { display: block; font-family: system-ui, sans-serif }
.diagram a:hover rect, .diagram a:focus rect { stroke-width: 4 }
.summary { color: #475569 }
.description { white-space: pre-wrap }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem }
td, th { border: 1px solid #cbd5e1; padding: .3rem .55rem; text-align: left; vertical-align: top }
th { background: #edf2f7 }
code { overflow-wrap: anywhere }
@media print {
  body { margin: 0; padding: 0 }
  .diagram { overflow: visible }
  .diagram svg { width: 100%; height: auto }
  tr, h2, h3, h4 { break-inside: avoid }
  h2, h3, h4 { break-after: avoid }
}
"""


class Notes(HTMLParser):
    """Keep the formatting supported by the block editor in an inert fragment."""

    tags = frozenset(
        "p div br b strong i em u s code pre blockquote ul ol li h1 h2 h3 h4 table tr td th".split()
    )
    blocked = frozenset("head script style iframe object template svg math".split())

    def __init__(self):
        super().__init__()
        self.parts = []
        self.open = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in self.blocked:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag == "br":
            self.parts.append("<br>")
            return
        if tag == "span":
            style = dict(attrs).get("style") or ""
            rules = {}
            for rule in style.split(";"):
                key, _, value = rule.partition(":")
                rules[key.strip().lower()] = value.strip().lower()
            decoration = rules.get("text-decoration", "").split()
            tags = tuple(
                name
                for name, used in (
                    ("strong", rules.get("font-weight") in {"bold", "600", "700", "800", "900"}),
                    ("em", rules.get("font-style") in {"italic", "oblique"}),
                    ("u", "underline" in decoration),
                    ("s", "line-through" in decoration),
                )
                if used
            )
        elif tag in self.tags:
            tags = (tag,)
        else:
            return
        self.open.append((tag, tags))
        self.parts.extend(f"<{name}>" for name in tags)

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        for index in range(len(self.open) - 1, -1, -1):
            if self.open[index][0] == tag:
                self.close_tags(index)
                return

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(escape(data))

    def close_tags(self, index):
        for _, tags in reversed(self.open[index:]):
            self.parts.extend(f"</{name}>" for name in reversed(tags))
        del self.open[index:]


def rich_notes(documentation):
    if not documentation.strip():
        return "<p>No documentation added.</p>"
    notes = Notes()
    notes.feed(documentation)
    notes.close()
    notes.close_tags(0)
    return "".join(notes.parts).strip() or "<p>No documentation added.</p>"


def table(headers, rows):
    head = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def link(anchor, label):
    return f'<a href="#{anchor}">{escape(label)}</a>'


def diagram(graph, anchors, prefix):
    if not anchors:
        return "<p>This canvas is empty.</p>"
    width, height, margin = 212, 58, 40
    positions = arranged_positions(graph, dict.fromkeys(anchors, (width, height)))
    right = int(max(x for x, _ in positions.values()) + width + margin)
    bottom = int(max(y for _, y in positions.values()) + height + margin)
    parts = [
        f'<div class="diagram"><svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Simplified block layout" viewBox="{-margin} {-margin} '
        f'{right + margin} {bottom + margin}" width="{right + margin}" height="{bottom + margin}">',
        f'<defs><marker id="{prefix}-arrow" markerWidth="6" markerHeight="6" refX="6" refY="3" '
        'orient="auto"><path d="M 0 0 L 6 3 L 0 6 Z" fill="#64748b"/></marker></defs>',
    ]
    for group in graph.groups:
        xs, ys = zip(*(positions[key] for key in group["members"]))
        x, y = min(xs) - 14, min(ys) - 27
        w, h = max(xs) - x + width + 14, max(ys) - y + height + 14
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            'fill="#eaf1f8" stroke="#94a3b8" stroke-dasharray="5 4"/>'
            f'<text x="{x + 9}" y="{y + 17}" font-size="12">{escape(group["name"])}</text>'
        )
    pairs = dict.fromkeys((wire.source, wire.target) for wire in graph.connections)
    for source, target in pairs:
        sx, sy = positions[source]
        tx, ty = positions[target]
        sx, sy, ty = sx + width, sy + height / 2, ty + height / 2
        mid = (sx + tx) / 2
        parts.append(
            f'<path d="M{sx},{sy} C{mid},{sy} {mid},{ty} {tx},{ty}" '
            f'fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#{prefix}-arrow)"/>'
        )
    for index, (key, anchor) in enumerate(anchors.items(), 1):
        block = graph.blocks[key]
        x, y = positions[key]
        category = block_type(block)
        name = block.name if len(block.name) <= 22 else block.name[:21] + "…"
        parts.append(
            f'<a href="#{anchor}"><title>{escape(block.name)}</title>'
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="7" '
            f'fill="white" stroke="{BLOCK_COLORS[category]}" stroke-width="2"/>'
            f'<text x="{x + 11}" y="{y + 23}" font-size="14" font-weight="bold">'
            f"{index}. {escape(name)}</text>"
            f'<text x="{x + 11}" y="{y + 43}" font-size="12" fill="#475569">'
            f"{escape(category)}</text></a>"
        )
    return "".join(parts) + "</svg></div>"


def block_details(block):
    parts = [
        f"<p><code>{escape(block.type_id)}</code> · "
        f"{escape(block.block_type or block.category)} · {escape(block.subtype)}</p>",
        "<h4>Description</h4>"
        f'<p class="description">{escape(block.description) or "No description added."}</p>',
        "<h4>Documentation</h4>",
        rich_notes(block.documentation),
    ]
    ports = [
        (
            direction,
            escape(port.name),
            escape(port.kind),
            "Required" if port.required else "Optional",
            "Exposed" if port.name in exposed else "Hidden",
        )
        for direction, entries, exposed in (
            ("Input", block.inputs, block.exposed_inputs),
            ("Output", block.output_ports, block.exposed_outputs),
        )
        for port in entries
    ]
    if ports:
        parts += [
            "<h4>Ports</h4>",
            table(("Direction", "Port", "Kind", "Presence", "Canvas"), ports),
        ]
    if block.parameters:
        rows = [
            (
                escape(name),
                f"<code>{escape(json.dumps(value, ensure_ascii=False, allow_nan=False))}</code>",
                escape(block.recipe_bindings.get(name, "")),
            )
            for name, value in block.parameters.items()
        ]
        parts += [
            "<h4>Applied parameters</h4>",
            table(("Parameter", "Value", "Recipe field"), rows),
        ]
    if block.recipe_model:
        parts.append(f"<p>Recipe model binding · {escape(block.recipe_model)}</p>")
    if block.ticket_source:
        parts.append(f"<p>Ticket mode · {escape(block.ticket_settings['mode'])}</p>")
    return parts


def canvas_sections(graph, prefix, title, interface=None):
    anchors = {key: f"{prefix}-{index}" for index, key in enumerate(graph.ordered(), 1)}
    parts = [
        '<section class="scope">',
        f"<h2>{escape(title)}</h2>",
        f'<p class="summary">{len(anchors)} blocks · {len(graph.connections)} connections</p>',
        diagram(graph, anchors, prefix),
    ]
    if interface is not None:
        bindings = [
            (
                direction,
                escape(name),
                link(anchors[binding.block], f"{graph.blocks[binding.block].name}.{binding.port}"),
            )
            for direction, entries in (
                ("Input", interface.input_bindings),
                ("Output", interface.output_bindings),
            )
            for name, binding in entries.items()
        ]
        if bindings:
            parts += [
                "<h3>Public terminals</h3>",
                table(("Direction", "Terminal", "Block port"), bindings),
            ]
    if graph.connections:
        rows = []
        for wire in graph.connections:
            source, target = graph.blocks[wire.source], graph.blocks[wire.target]
            out = next(port for port in source.output_ports if port.name == wire.output)
            inp = next(port for port in target.inputs if port.name == wire.input)
            rows.append(
                (
                    link(anchors[wire.source], f"{source.name}.{wire.output}"),
                    link(anchors[wire.target], f"{target.name}.{wire.input}"),
                    escape(connection_mode(out.kind, inp.kind) or "unknown"),
                )
            )
        parts += ["<h3>Connections</h3>", table(("Source", "Target", "Rule"), rows)]
    if anchors:
        parts.append("<h3>Block index</h3><ol>")
        parts.extend(
            f"<li>{link(anchor, graph.blocks[key].name)}</li>" for key, anchor in anchors.items()
        )
        parts.append("</ol>")
    for index, (key, anchor) in enumerate(anchors.items(), 1):
        block = graph.blocks[key]
        parts.append(f'<section class="block" id="{anchor}"><h3>{index}. {escape(block.name)}</h3>')
        parts.extend(block_details(block))
        interface = block.inspection if isinstance(block, ParallelInspect) else block
        if isinstance(interface, ToolBlock):
            parts.extend(canvas_sections(interface.graph, anchor, "Internal layout", interface))
        parts.append("</section>")
    return [*parts, "</section>"]


def render_program_documentation(graph, title):
    title = escape(title)
    settings = escape(
        f"Run profile · {graph.run_profile.name} | Branch workers · {graph.branch_workers}"
        f" | Graphics default · {graph.graphics_mode}"
    )
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">",
        f"<title>{title} · Program documentation</title><style>{STYLES}</style></head><body>",
        f"<header><h1>{title}</h1><p>OpenCV Dataflow program documentation</p>",
        f'<p class="summary">{settings}</p></header>',
        *canvas_sections(graph, "job", "Program layout"),
        "</body></html>",
    ]
    return "\n".join(parts) + "\n"


def export_program_documentation(graph, path, title="Untitled program"):
    path = Path(path)
    if path.suffix.lower() not in (".html", ".htm"):
        raise ValueError("Choose an .html file for program documentation")
    content = render_program_documentation(graph, title)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".program-doc-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8"))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
