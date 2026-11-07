"""Route canvas connections around nearby block bodies."""

from itertools import pairwise

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainterPath, QPainterPathStroker

CLEARANCE = 10
LEAD = 24
CORNER_RADIUS = 8
MAX_LANES = 24


def curve(start, end):
    path = QPainterPath(start)
    span = end.x() - start.x()
    distance = max(70, abs(span) * 0.5)
    if span > 0:
        distance = min(distance, span)
    path.cubicTo(start + QPointF(distance, 0), end - QPointF(distance, 0), end)
    return path


def _padded(rect, margin):
    return rect.adjusted(-margin, -margin, margin, margin)


def _segment_hits(start, end, box):
    return (
        min(start.x(), end.x()) < box.right()
        and max(start.x(), end.x()) > box.left()
        and min(start.y(), end.y()) < box.bottom()
        and max(start.y(), end.y()) > box.top()
    )


def _rounded_path(points):
    points = [points[0], *(b for a, b in pairwise(points) if a != b)]
    path = QPainterPath(points[0])
    for previous, corner, following in zip(points, points[1:], points[2:]):
        incoming, outgoing = corner - previous, following - corner
        into = min(0.5, CORNER_RADIUS / incoming.manhattanLength())
        out = min(0.5, CORNER_RADIUS / outgoing.manhattanLength())
        path.lineTo(corner - into * incoming)
        path.quadTo(corner, corner + out * outgoing)
    path.lineTo(points[-1])
    return path


def route_wire(start, end, source, target, nearby_blocks):
    """Keep a clear forward curve. Otherwise try nearby horizontal detour lanes."""
    def obstacles(bounds):
        return [_padded(box, CLEARANCE) for box in nearby_blocks(_padded(bounds, CLEARANCE))]

    direct = curve(start, end)
    forward = start.x() < end.x()
    direct_hits = 0
    if forward:
        boxes = obstacles(direct.boundingRect())
        if boxes:
            stroke = QPainterPathStroker().createStroke(direct)
            direct_hits = sum(stroke.intersects(box) for box in boxes)
        if not direct_hits:
            return direct

    source, target = _padded(source, CLEARANCE), _padded(target, CLEARANCE)
    exit_point, entry_point = start + QPointF(LEAD, 0), end - QPointF(LEAD, 0)
    corridor = QRectF(start, end).normalized().united(QRectF(exit_point, entry_point).normalized())
    # A horizontal wire has zero height. Expand before querying the scene index.
    nearby = obstacles(corridor.adjusted(0, -LEAD, 0, LEAD))
    y0, y1 = start.y(), end.y()
    lanes = {y0, y1, (y0 + y1) / 2}
    for box in (*nearby, source, target):
        lanes.update((box.top(), box.bottom()))

    def lane_order(y):
        return abs(y - y0) + abs(y - y1), abs(y - (y0 + y1) / 2), y

    lanes = sorted(lanes, key=lane_order)[:MAX_LANES]
    # Lanes may extend above or below the original search if a block is tall.
    search = QRectF(corridor.left(), min(lanes), corridor.width(), max(lanes) - min(lanes))
    boxes = obstacles(search)
    best = None
    fewest = direct_hits if forward else len(boxes) + 3
    for y in lanes:
        points = [
            start, exit_point, QPointF(exit_point.x(), y), QPointF(entry_point.x(), y),
            entry_point, end,
        ]
        legs = list(pairwise(points))
        hits = sum(any(_segment_hits(a, b, box) for a, b in legs) for box in boxes)
        hits += any(_segment_hits(a, b, source) for a, b in legs[1:])
        hits += any(_segment_hits(a, b, target) for a, b in legs[:-1])
        if hits < fewest:
            fewest, best = hits, points
            if not hits:
                break
    return direct if best is None else _rounded_path(best)
