"""Graphics View node canvas with direction-independent wiring."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
)


def pixmap(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        self.setPos(220 if output else 0, 62 + row * 25)
        colors = {"image": "#51c7db", "number": "#dca65e", "data": "#b497ea", "any": "#c2cbd5"}
        self.setBrush(QColor(colors.get(port.kind, "#c2cbd5")))
        self.setPen(QPen(QColor("#17202c"), 2))
        self.setToolTip(f"{port.name} · {port.kind} · {'output' if output else 'input'}")
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setZValue(3)

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
        self.sockets = {}
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPos(*block.position)
        for output, ports, exposed in (
            (False, block.inputs, block.exposed_inputs),
            (True, block.outputs, block.exposed_outputs),
        ):
            row = 0
            for port in ports:
                if port.name in exposed:
                    self.sockets[output, port.name] = Socket(self, port, output, row)
                    row += 1
        self.rows = max(len(block.exposed_inputs), len(block.exposed_outputs), 1)
        self.height = 100 + self.rows * 25 + (100 if block.thumbnail else 0)

    def boundingRect(self):
        return QRectF(-9, -2, 238, self.height + 5)

    def mouseDoubleClickEvent(self, event):
        if not self.scene().locked:
            self.scene().open_block.emit(self.block)
        event.accept()

    def paint(self, painter, option, widget=None):
        b = self.block
        base = {
            "idle": "#354257",
            "pass": "#26835d",
            "fail": "#ae6430",
            "error": "#b14e5b",
            "skipped": "#646377",
        }[b.state]
        status = QColor(base)
        if b.last_result and b.last_result.score is not None:
            status = status.lighter(75 + int(50 * b.last_result.score))
        painter.setPen(
            QPen(
                QColor("#67d1e1" if self.isSelected() else "#43536b"), 2 if self.isSelected() else 1
            )
        )
        painter.setBrush(QColor("#202c3d"))
        painter.drawRoundedRect(QRectF(0, 0, 220, self.height), 9, 9)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(status)
        painter.drawRoundedRect(QRectF(1, 1, 218, 39), 8, 8)
        painter.setPen(QColor("#f4f7fb"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QRectF(12, 0, 196, 39),
            Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(b.name, Qt.TextElideMode.ElideRight, 193),
        )
        font.setBold(False)
        painter.setFont(font)
        for (output, name), socket in self.sockets.items():
            rect = QRectF(12 if not output else 110, socket.pos().y() - 10, 98, 20)
            painter.setPen(QColor("#c4cfdd"))
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignRight if output else Qt.AlignmentFlag.AlignLeft,
                painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, 94),
            )
        y = 72 + self.rows * 25
        if b.thumbnail:
            images = {**b.last_output_images, **b.last_graphics}
            image = images.get(b.preview_output, next(iter(images.values()), None))
            painter.setBrush(QColor("#152030"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(12, y, 196, 90), 5, 5)
            if image is not None:
                thumb = pixmap(image).scaled(
                    196,
                    90,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(int(110 - thumb.width() / 2), int(y), thumb)
            else:
                painter.setPen(QColor("#74859b"))
                painter.drawText(
                    QRectF(12, y, 196, 90), Qt.AlignmentFlag.AlignCenter, "Run to preview"
                )
            y += 99
        painter.setPen(QColor("#a6b6cb"))
        label = b.state.upper()
        if b.last_result and b.last_result.score is not None:
            label += f"  {b.last_result.score:.0%}"
        if b.state != "idle":
            label += f"  ·  {b.elapsed_ms:.1f} ms"
        painter.drawText(QRectF(12, y - 3, 196, 22), Qt.AlignmentFlag.AlignLeft, label)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.block.position = [self.pos().x(), self.pos().y()]
            if self.scene():
                self.scene().update_wires()
                self.scene().edited.emit()
        return super().itemChange(change, value)


class Wire(QGraphicsPathItem):
    def __init__(self, connection, source, target):
        super().__init__()
        self.connection, self.source, self.target = connection, source, target
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)
        self.refresh()

    def refresh(self):
        self.setPath(curve(self.source.scenePos(), self.target.scenePos()))
        self.setPen(QPen(QColor("#eab76a" if self.isSelected() else "#4b9fb5"), 3))

    def paint(self, painter, option, widget=None):
        self.setPen(QPen(QColor("#eab76a" if self.isSelected() else "#4b9fb5"), 3))
        super().paint(painter, option, widget)


class CanvasScene(QGraphicsScene):
    selected_block = Signal(object)
    open_block = Signal(object)
    edited = Signal()
    warning = Signal(str)

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.nodes = {}
        self.wires = []
        self.locked = False
        self.drag_socket = None
        self.drag_path = None
        self.setSceneRect(-1500, -1200, 5000, 3200)
        self.selectionChanged.connect(self.on_selection)
        self.rebuild()

    def rebuild(self, selected=None):
        self.drag_socket = None
        self.drag_path = None
        self.clear()
        self.nodes, self.wires = {}, []
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

    def on_selection(self):
        node = next((n for n in self.selectedItems() if isinstance(n, Node)), None)
        self.selected_block.emit(node.block if node else None)

    def update_wires(self):
        for wire in self.wires:
            wire.refresh()

    def start_wire(self, socket):
        self.drag_socket = socket
        self.drag_path = QGraphicsPathItem()
        self.drag_path.setPen(QPen(QColor("#86e5ee"), 2, Qt.PenStyle.DashLine))
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
        self.graph.invalidate()
        self.rebuild()
        self.edited.emit()

    def set_locked(self, locked):
        self.locked = locked
        for node in self.nodes.values():
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

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        painter.setPen(QPen(QColor("#253247"), 1))
        left, top = int(rect.left()) // 25 * 25, int(rect.top()) // 25 * 25
        for x in range(left, int(rect.right()), 25):
            for y in range(top, int(rect.bottom()), 25):
                painter.drawPoint(QPointF(x, y))

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if 0.2 < self.transform().m11() * factor < 3:
            self.scale(factor, factor)

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
        else:
            super().mouseReleaseEvent(event)

    def fit_graph(self):
        if self.scene().nodes:
            bounds = QRectF()
            for node in self.scene().nodes.values():
                bounds = bounds.united(node.sceneBoundingRect())
            self.fitInView(bounds.adjusted(-45, -60, 45, 60), Qt.AspectRatioMode.KeepAspectRatio)
