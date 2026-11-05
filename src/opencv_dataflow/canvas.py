"""Graphics View node canvas with direction-independent wiring."""

from __future__ import annotations

import json
from html import escape

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from .images import preview_rgb
from .port_types import connection_mode
from .visual_style import (
    BLOCK_COLORS,
    BLOCK_STATE_COLORS,
    NODE_SIZES,
    PORT_COLORS,
    block_type,
    connection_type,
    port_type,
)
from .workspace import arranged_positions


def pixmap(image):
    rgb = preview_rgb(image)
    rgb = np.ascontiguousarray(rgb)
    return QPixmap.fromImage(
        QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
    )


def curve(start, end):
    path = QPainterPath(start)
    distance = max(70, abs(end.x() - start.x()) * 0.5)
    path.cubicTo(start + QPointF(distance, 0), end - QPointF(distance, 0), end)
    return path


class Socket(QGraphicsEllipseItem):
    def __init__(self, node, port, output, row):
        super().__init__(-7, -7, 14, 14, node)
        self.node, self.port, self.output = node, port, output
        self.row = row
        self.visual_type = port_type(node.block, port, output)
        self.setBrush(QColor(PORT_COLORS[self.visual_type]))
        self.setPen(QPen(QColor("#17202c"), 2))
        self.refresh_tooltip()
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setZValue(3)

    def refresh_tooltip(self):
        block, port = self.node.block, self.port
        direction = "output" if self.output else "input"
        heading = escape(f"{port.name} · {port.kind} · {self.visual_type} color · {direction}")
        values = block.last_inputs
        if self.output:
            values = block.last_result.outputs if block.last_result is not None else {}
        detail = ""
        if port.name in values:
            value = values[port.name]
            if isinstance(value, str):
                text = value[:160] + ("…" if len(value) > 160 else "")
                detail = f"<br>Value: {escape(json.dumps(text, ensure_ascii=False))}"
            elif value is None or isinstance(
                value, (bool, int, float, np.bool_, np.integer, np.floating)
            ):
                detail = f"<br>Value: {escape(str(value))}"
        help_text = "" if self.output else block.input_help.get(port.name, "")
        help_line = f"<br>{escape(help_text)}" if help_text else ""
        self.setToolTip(f"<qt>{heading}{help_line}{detail}</qt>")

    def mousePressEvent(self, event):
        if not self.scene().locked:
            self.scene().start_wire(self)
        event.accept()

    def mouseMoveEvent(self, event):
        self.scene().move_wire(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        scene = self.scene()
        target = next(
            (
                item
                for item in scene.items(event.scenePos())
                if isinstance(item, Socket) and item is not self
            ),
            None,
        )
        scene.finish_wire(target)
        event.accept()


class Node(QGraphicsItem):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.category = block_type(block)
        self.border_color = QColor(BLOCK_COLORS[self.category])
        heading = escape(f"{block.name} · {self.category}")
        description = "<br>".join(escape(line) for line in block.description.split("\n"))
        detail = f"<br><br>{description}" if description else ""
        self._tooltip = f"{heading}{detail}<br>Border: tool category · header: run status"
        self._draft_issue = ""
        self.setToolTip(f"<qt>{self._tooltip}</qt>")
        self.setAcceptHoverEvents(True)
        self.sockets = {}
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPos(*block.position)
        for output, ports, exposed in (
            (False, block.inputs, block.exposed_inputs),
            (True, block.output_ports, block.exposed_outputs),
        ):
            row = 0
            for port in ports:
                if port.name in exposed:
                    self.sockets[output, port.name] = Socket(self, port, output, row)
                    row += 1
        self.rows = max(len(block.exposed_inputs), len(block.exposed_outputs), 1)
        self.update_size()

    def update_size(self):
        self.width, spacing, thumbnail_height = NODE_SIZES[self.block.node_size]
        for socket in self.sockets.values():
            socket.setPos(self.width if socket.output else 0, 62 + socket.row * spacing)
        bottom = 72 + self.rows * spacing
        thumbnail_height = thumbnail_height if self.block.thumbnail else 0
        self.thumbnail_rect = QRectF(12, bottom, self.width - 24, thumbnail_height)
        self.status_y = bottom + (thumbnail_height + 9 if thumbnail_height else 0)
        self.height = self.status_y + 28

    def set_size(self, size):
        if size == self.block.node_size:
            return
        if size not in NODE_SIZES:
            raise ValueError("Invalid block size")
        self.prepareGeometryChange()
        self.block.node_size = size
        self.update_size()
        if self.scene():
            self.scene().geometry_changed()
        self.update()

    def boundingRect(self):
        return QRectF(-9, -2, self.width + 18, self.height + 5)

    def mousePressEvent(self, event):
        # Rubber-band selection can include both a visual container and its nodes.
        # Dragging a node moves the selected nodes only, avoiding a second group delta.
        for group in self.scene().groups.values():
            group.setSelected(False)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self.scene().locked:
            self.scene().open_block.emit(self.block)
        event.accept()

    def issue_badges(self):
        badges = []
        invalid = f"Invalid configuration · {self._draft_issue}" if self._draft_issue else ""
        if self.block.state == "error" and self.block.error_kind == "invalid":
            runtime = f"Invalid data · {self.block.error}"
            invalid = f"{invalid}\n{runtime}" if invalid else runtime
        if invalid:
            badges.append(("!", "#ffc575", invalid))
        if self.block.state == "error" and self.block.error_kind == "exception":
            badges.append(("×", "#ff999b", f"Exception · {self.block.error}"))
        return badges

    def refresh_issue_tooltip(self):
        details = "".join(
            "<br>" + escape(detail).replace("\n", "<br>")
            for _, _, detail in self.issue_badges()
        )
        tooltip = f"<qt>{self._tooltip}{details}</qt>"
        if self.toolTip() != tooltip:
            self.setToolTip(tooltip)

    def set_draft_issue(self, message):
        if message != self._draft_issue:
            self._draft_issue = message
            self.refresh_issue_tooltip()
            self.update()

    def hoverEnterEvent(self, event):
        self.refresh_issue_tooltip()
        super().hoverEnterEvent(event)

    def paint(self, painter, option, widget=None):
        b = self.block
        status = QColor(BLOCK_STATE_COLORS[b.state])
        if b.last_result and b.last_result.score is not None:
            status = status.lighter(75 + int(50 * b.last_result.score))
        if self.isSelected():
            painter.setPen(QPen(QColor("#f4f7fb"), 5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(0, 0, self.width, self.height), 9, 9)
        painter.setPen(QPen(self.border_color, 2))
        painter.setBrush(QColor("#202c3d"))
        painter.drawRoundedRect(QRectF(0, 0, self.width, self.height), 9, 9)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(status)
        painter.drawRoundedRect(QRectF(1, 1, self.width - 2, 39), 8, 8)
        painter.setPen(QColor("#f4f7fb"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        content_width = self.width - 24
        badges = self.issue_badges()
        title_width = content_width - 22 * len(badges)
        painter.drawText(
            QRectF(12, 0, title_width, 39),
            Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                b.name, Qt.TextElideMode.ElideRight, title_width - 3
            ),
        )
        for index, (symbol, color, _) in enumerate(badges):
            badge = QRectF(12 + title_width + index * 22, 10, 18, 18)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(badge)
            painter.setPen(QColor("#202c3d"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, symbol)
        font.setBold(False)
        painter.setFont(font)
        port_width = content_width // 2
        for (output, name), socket in self.sockets.items():
            rect = QRectF(self.width / 2 if output else 12, socket.pos().y() - 10, port_width, 20)
            painter.setPen(QColor("#c4cfdd"))
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignRight if output else Qt.AlignmentFlag.AlignLeft,
                painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, port_width - 4),
            )
        thumbnail = self.thumbnail_rect
        if not thumbnail.isEmpty():
            images = {**b.last_output_images, **b.last_graphics}
            image = images.get(b.preview_output, next(iter(images.values()), None))
            painter.setBrush(QColor("#152030"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(thumbnail, 5, 5)
            if image is not None:
                thumb = pixmap(image).scaled(
                    thumbnail.size().toSize(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                center = thumbnail.center()
                painter.drawPixmap(
                    int(center.x() - thumb.width() / 2), int(center.y() - thumb.height() / 2), thumb
                )
            else:
                painter.setPen(QColor("#74859b"))
                painter.drawText(
                    thumbnail, Qt.AlignmentFlag.AlignCenter, "Run to preview"
                )
        painter.setPen(QColor("#a6b6cb"))
        label = b.state.upper()
        if b.last_result and b.last_result.score is not None:
            label += f"  {b.last_result.score:.0%}"
        if b.state != "idle":
            label += f"  ·  {b.elapsed_ms:.1f} ms"
        painter.drawText(
            QRectF(12, self.status_y - 3, content_width, 22), Qt.AlignmentFlag.AlignLeft, label
        )

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.block.position = [self.pos().x(), self.pos().y()]
            if self.scene():
                self.scene().geometry_changed()
        return super().itemChange(change, value)


class Wire(QGraphicsPathItem):
    def __init__(self, connection, source, target):
        super().__init__()
        self.connection, self.source, self.target = connection, source, target
        self.visual_type = connection_type(source.visual_type, target.visual_type)
        self.color = QColor(PORT_COLORS[self.visual_type])
        mode = connection_mode(source.port.kind, target.port.kind)
        detail = ""
        if mode is not None and mode != "direct":
            detail = f" · {source.port.kind} → {target.port.kind} · {mode}"
        self.setToolTip(f"{source.port.name} → {target.port.name} · {self.visual_type}{detail}")
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)
        self.refresh()

    def refresh(self):
        self.setPath(curve(self.source.scenePos(), self.target.scenePos()))
        pen = QPen(self.color, 3)
        pen.setCosmetic(True)
        self.setPen(pen)

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.isSelected():
            halo = QPen(QColor("#f4f7fb"), 7)
            halo.setCosmetic(True)
            painter.setPen(halo)
            painter.drawPath(self.path())
        painter.setPen(self.pen())
        painter.drawPath(self.path())
        painter.restore()


class CanvasScene(QGraphicsScene):
    selected_block = Signal(object)
    open_block = Signal(object)
    edited = Signal()
    layout_changed = Signal()
    warning = Signal(str)

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.nodes = {}
        self.wires = []
        self.groups = {}
        self._moving_nodes = False
        self.locked = False
        self.drag_socket = None
        self.drag_path = None
        self.setSceneRect(-1500, -1200, 5000, 3200)
        self.selectionChanged.connect(self.on_selection)
        self.rebuild()

    def helpEvent(self, event):
        for item in self.items(event.scenePos()):
            if isinstance(item, Socket):
                item.refresh_tooltip()
                break
            if item.toolTip():
                break
        super().helpEvent(event)

    def rebuild(self, selected=None):
        self.drag_socket = None
        self.drag_path = None
        self.clear()
        self.nodes, self.wires, self.groups = {}, [], {}
        for b in self.graph.blocks.values():
            node = Node(b)
            self.addItem(node)
            self.nodes[b.id] = node
            if b.id == selected:
                node.setSelected(True)
        for c in self.graph.connections:
            wire = Wire(
                c,
                self.nodes[c.source].sockets[True, c.output],
                self.nodes[c.target].sockets[False, c.input],
            )
            self.addItem(wire)
            self.wires.append(wire)

        for group in self.graph.groups:
            item = NodeGroup(group)
            self.addItem(item)
            self.groups[group["id"]] = item
            item.refresh()
        self.expand_workspace()
        self.set_locked(self.locked)

    def expand_workspace(self):
        bounds = self.itemsBoundingRect().adjusted(-1000, -1000, 1000, 1000)
        for view in self.views():
            bounds = bounds.united(view.mapToScene(view.viewport().rect()).boundingRect())
        self.setSceneRect(bounds.united(QRectF(-1500, -1200, 5000, 3200)))

    def geometry_changed(self):
        if self._moving_nodes:
            return
        self.update_wires()
        for group in self.groups.values():
            group.refresh()
        self.expand_workspace()
        self.layout_changed.emit()

    def auto_arrange(self):
        if self.locked or not self.nodes:
            return False
        sizes = {
            key: (node.boundingRect().width(), node.boundingRect().height())
            for key, node in self.nodes.items()
        }
        positions = arranged_positions(self.graph, sizes)
        moved = [
            (self.nodes[key], QPointF(*point))
            for key, point in positions.items()
            if self.nodes[key].pos() != QPointF(*point)
        ]
        if not moved:
            return False
        self._moving_nodes = True
        try:
            for node, point in moved:
                node.setPos(point)
        finally:
            self._moving_nodes = False
        self.geometry_changed()
        return True

    def group_selected(self, name):
        from .workspace import add_group

        if self.locked:
            return
        members = [n.block.id for n in self.selectedItems() if isinstance(n, Node)]
        if not members:
            raise ValueError("Select nodes to group first")
        group = add_group(self.graph, members, name)
        self.rebuild()
        self.groups[group["id"]].setSelected(True)
        self.layout_changed.emit()
        return group

    def ungroup_selected(self):
        if self.locked:
            return
        selected = self.selectedItems()
        ids = {g.data["id"] for g in selected if isinstance(g, NodeGroup)}
        members = {n.block.id for n in selected if isinstance(n, Node)}
        self.graph.groups = [
            g
            for g in self.graph.groups
            if g["id"] not in ids and not members.intersection(g["members"])
        ]
        self.rebuild()
        self.layout_changed.emit()

    def on_selection(self):
        node = next((n for n in self.selectedItems() if isinstance(n, Node)), None)
        self.selected_block.emit(node.block if node else None)

    def update_wires(self):
        for wire in self.wires:
            wire.refresh()

    def start_wire(self, socket):
        self.drag_socket = socket
        self.drag_path = QGraphicsPathItem()
        pen = QPen(QColor(PORT_COLORS[socket.visual_type]), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.drag_path.setPen(pen)
        self.addItem(self.drag_path)

    def move_wire(self, position):
        if self.drag_path:
            start = self.drag_socket.scenePos()
            self.drag_path.setPath(
                curve(start, position) if self.drag_socket.output else curve(position, start)
            )

    def finish_wire(self, target):
        start = self.drag_socket
        if self.drag_path:
            self.removeItem(self.drag_path)
        self.drag_path, self.drag_socket = None, None
        if start is None or target is None:
            return
        try:
            if start.output == target.output:
                raise ValueError("Connect an output to an input")
            source, dest = (start, target) if start.output else (target, start)
            self.graph.connect(
                source.node.block.id, source.port.name, dest.node.block.id, dest.port.name
            )
            self.rebuild(dest.node.block.id)
            self.edited.emit()
        except (ValueError, KeyError) as exc:
            self.warning.emit(str(exc))

    def delete_selected(self):
        if self.locked:
            return
        selected = list(self.selectedItems())
        for item in selected:
            if isinstance(item, Wire) and item.connection in self.graph.connections:
                self.graph.connections.remove(item.connection)
            elif isinstance(item, Node) and item.block.id in self.graph.blocks:
                self.graph.remove(item.block.id)
        topology_changed = any(isinstance(i, (Node, Wire)) for i in selected)
        group_ids = {i.data["id"] for i in selected if isinstance(i, NodeGroup)}
        self.graph.groups = [g for g in self.graph.groups if g["id"] not in group_ids]
        if topology_changed:
            self.graph.invalidate()
        self.rebuild()
        (self.edited if topology_changed else self.layout_changed).emit()

    def set_locked(self, locked):
        self.locked = locked
        for node in [*self.nodes.values(), *self.groups.values()]:
            node.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not locked)


class CanvasView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#111b29"))
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumWidth(380)
        self.pan_start = None
        self._restoring = False
        self.overview = Overview(self)
        scene.layout_changed.connect(self.overview.update)
        scene.changed.connect(self.overview.update)
        for scrollbar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            scrollbar.valueChanged.connect(lambda value: self.overview.update())
            scrollbar.sliderReleased.connect(self.remember_view)
            scrollbar.actionTriggered.connect(
                lambda action: QTimer.singleShot(0, self.remember_view)
            )

    def remember_view(self, *args):
        self.overview.update()
        if self._restoring or not self.isVisible():
            return
        center = self.mapToScene(self.viewport().rect().center())
        self.scene().graph.viewport = {
            "center": [center.x(), center.y()],
            "zoom": max(0.005, min(3.0, self.transform().m11())),
        }
        self.scene().layout_changed.emit()

    def restore_view(self):
        saved = self.scene().graph.viewport
        self._restoring = True
        try:
            if saved:
                self.setTransform(QTransform.fromScale(saved["zoom"], saved["zoom"]))
                center = QPointF(*saved["center"])
                half_width = self.viewport().width() / (2 * saved["zoom"]) + 1000
                half_height = self.viewport().height() / (2 * saved["zoom"]) + 1000
                self.scene().setSceneRect(
                    self.sceneRect().united(
                        QRectF(
                            center.x() - half_width,
                            center.y() - half_height,
                            2 * half_width,
                            2 * half_height,
                        )
                    )
                )
                self.centerOn(center)
            else:
                self.fit_graph(remember=False)
        finally:
            self._restoring = False
        self.overview.update()

    def resizeEvent(self, event):
        self._restoring = True
        try:
            super().resizeEvent(event)
        finally:
            self._restoring = False
        self.overview.move(max(0, self.viewport().width() - self.overview.width() - 12), 12)
        self.overview.update()

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        painter.setPen(QPen(QColor("#253247"), 1))
        spacing = 25
        while spacing * self.transform().m11() < 15:
            spacing *= 5
        left, top = int(rect.left()) // spacing * spacing, int(rect.top()) // spacing * spacing
        for x in range(left, int(rect.right()), spacing):
            for y in range(top, int(rect.bottom()), spacing):
                painter.drawPoint(QPointF(x, y))

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if 0.005 <= self.transform().m11() * factor <= 3:
            self.scale(factor, factor)
            self.remember_view()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.scene().delete_selected()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan_start is not None:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = None
            self.unsetCursor()
            self.remember_view()
        else:
            super().mouseReleaseEvent(event)

    def fit_graph(self, remember=True):
        if self.scene().nodes:
            bounds = QRectF()
            for node in self.scene().nodes.values():
                bounds = bounds.united(node.sceneBoundingRect())
            previous = self._restoring
            self._restoring = True
            try:
                self.fitInView(
                    bounds.adjusted(-45, -60, 45, 60), Qt.AspectRatioMode.KeepAspectRatio
                )
                zoom = max(0.005, min(3.0, self.transform().m11()))
                self.setTransform(QTransform.fromScale(zoom, zoom))
                self.centerOn(bounds.center())
            finally:
                self._restoring = previous
            if remember:
                self.remember_view()


class NodeGroup(QGraphicsItem):
    """Visual container; absolute node coordinates and processing remain independent."""

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.rect = QRectF()
        self.setZValue(-2)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    def refresh(self):
        bounds = QRectF()
        for key in self.data["members"]:
            bounds = bounds.united(self.scene().nodes[key].sceneBoundingRect())
        self.prepareGeometryChange()
        self.rect = bounds.adjusted(-22, -42, 22, 22).translated(-self.pos())
        self.update()

    def boundingRect(self):
        return self.rect.adjusted(-3, -3, 3, 3)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        for group in self.scene().groups.values():
            if group.isSelected():
                for key in group.data["members"]:
                    self.scene().nodes[key].setSelected(False)

    def shape(self):
        outline = QPainterPath()
        outline.addRect(self.rect)
        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        path = stroker.createStroke(outline)
        path.addRect(QRectF(self.rect.left(), self.rect.top(), self.rect.width(), 32))
        return path

    def paint(self, painter, option, widget=None):
        painter.setPen(
            QPen(QColor("#73d5e2" if self.isSelected() else "#66829f"), 2, Qt.PenStyle.DashLine)
        )
        painter.setBrush(QColor(70, 100, 140, 35))
        painter.drawRoundedRect(self.rect, 8, 8)
        painter.setPen(QColor("#c6d7eb"))
        painter.drawText(
            self.rect.adjusted(12, 3, -12, 0), Qt.AlignmentFlag.AlignTop, self.data["name"]
        )

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            delta = value - self.pos()
            scene = self.scene()
            scene._moving_nodes = True
            try:
                for key in self.data["members"]:
                    node = scene.nodes[key]
                    node.setPos(node.pos() + delta)
            finally:
                scene._moving_nodes = False
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            self.scene().geometry_changed()
        return result


class Overview(QWidget):
    """Cheap workspace map: no image previews or full-scene rendering."""

    def __init__(self, view):
        super().__init__(view.viewport())
        self.view = view
        self.setFixedSize(200, 140)
        self.setToolTip("Workspace overview · click or drag to navigate")
        self.world = QRectF()
        self.ratio = 1.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(17, 27, 41, 235))
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        self.world = (
            self.view.scene().itemsBoundingRect().united(visible).adjusted(-50, -50, 50, 50)
        )
        self.ratio = min(
            (self.width() - 12) / max(1, self.world.width()),
            (self.height() - 12) / max(1, self.world.height()),
        )
        painter.translate(6, 6)
        painter.scale(self.ratio, self.ratio)
        painter.translate(-self.world.left(), -self.world.top())
        for group in self.view.scene().groups.values():
            painter.fillRect(group.sceneBoundingRect(), QColor("#31445c"))
        for node in self.view.scene().nodes.values():
            painter.fillRect(node.sceneBoundingRect(), node.border_color)
        painter.setPen(QPen(QColor("#f3c572"), 2 / self.ratio))
        painter.setBrush(QColor(243, 197, 114, 25))
        painter.drawRect(visible)

    def navigate(self, position):
        point = QPointF(
            self.world.left() + (position.x() - 6) / self.ratio,
            self.world.top() + (position.y() - 6) / self.ratio,
        )
        self.view.centerOn(point)
        self.view.remember_view()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.navigate(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.navigate(event.position())
