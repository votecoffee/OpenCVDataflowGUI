"""Modeless ticket lookup over the thread-safe runtime result consumer."""

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QLineEdit,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from .tickets import ticket_key


class TicketResultsWindow(QDialog):
    def __init__(self, consumer, parent=None):
        super().__init__(parent)
        self.consumer = consumer
        self.revision = None
        self.setWindowTitle("Ticket results")
        self.resize(1100, 650)
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Search ticket value, kind, session, source or current status"
        )
        layout.addWidget(self.search)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        split = QSplitter()
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["Ticket", "Kind", "Session", "Status", "Events", "Sources"])
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.events = QTreeWidget()
        self.events.setHeaderLabels(["UTC time", "Tool", "Result", "Score", "Event", "Message"])
        event_panel = QSplitter(Qt.Orientation.Vertical)
        event_panel.addWidget(self.events)
        event_panel.addWidget(self.details)
        split.addWidget(self.table)
        split.addWidget(event_panel)
        layout.addWidget(split)
        self.rows = {}
        self.search.textChanged.connect(self.refresh)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.events.itemSelectionChanged.connect(self.show_event)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        consumer = self.consumer()
        revision = (id(consumer), consumer.revision(), self.search.text())
        if revision == self.revision:
            return
        self.revision = revision
        snapshot = consumer.snapshot()
        selected = self.table.currentItem()
        key = selected.data(0, Qt.ItemDataRole.UserRole) if selected else None
        self.table.blockSignals(True)
        self.table.clear()
        self.rows = {}
        query = self.search.text().casefold()
        for row in reversed(snapshot["tickets"]):
            ticket = row["ticket"]
            fields = [
                str(ticket["value"]),
                ticket["kind"],
                ticket["session"],
                row["status"],
                str(row["event_count"]),
                ", ".join(row["sources"]),
            ]
            if query and query not in " ".join(fields).casefold():
                continue
            identity = json.dumps(ticket_key(ticket))
            self.rows[identity] = row
            item = QTreeWidgetItem(self.table, fields)
            item.setData(0, Qt.ItemDataRole.UserRole, identity)
            if identity == key:
                self.table.setCurrentItem(item)
        self.table.blockSignals(False)
        if self.table.currentItem() is None and self.table.topLevelItemCount():
            self.table.setCurrentItem(self.table.topLevelItem(0))
        self.summary.setText(
            f"{len(snapshot['tickets'])} retained tickets · "
            f"{snapshot['evicted_tickets']} evicted · "
            "Runtime history only; intermediate results do not finalize a ticket."
        )
        self.show_selected()

    def show_selected(self):
        item = self.table.currentItem()
        row = self.rows.get(item.data(0, Qt.ItemDataRole.UserRole)) if item else None
        selected = self.events.currentItem()
        selected_id = selected.data(0, Qt.ItemDataRole.UserRole)["id"] if selected else None
        self.events.blockSignals(True)
        self.events.clear()
        for event in row["events"] if row else []:
            verdict = (
                "Pass"
                if event["passed"] is True
                else "Fail"
                if event["passed"] is False
                else "Event"
            )
            if event["kind"] == "input_rejected":
                verdict = "Rejected"
            elif event["kind"] == "coordination_late":
                verdict = "Ignored late input"
            elif event["kind"] == "error":
                verdict = "Error"
            elif event["final_judgment"] is not None:
                verdict = "Final PASS" if event["final_judgment"] else "Final FAIL"
            elif event["passed"] and event["chain_passed"] is False:
                verdict = "Pass (upstream failed)"
            fields = [
                event["timestamp"],
                event["source"].get("name", event["source"]["id"]),
                verdict,
                f"{event['score']:.1%}" if event["score"] is not None else "",
                event["kind"],
                event["message"],
            ]
            entry = QTreeWidgetItem(self.events, fields)
            entry.setData(0, Qt.ItemDataRole.UserRole, event)
            if event["id"] == selected_id:
                self.events.setCurrentItem(entry)
        self.events.blockSignals(False)
        if self.events.currentItem() is None and self.events.topLevelItemCount():
            self.events.setCurrentItem(
                self.events.topLevelItem(self.events.topLevelItemCount() - 1)
            )
        self.show_event()

    def show_event(self):
        item = self.events.currentItem()
        event = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        self.details.setPlainText(
            json.dumps(event, indent=2, ensure_ascii=False)
            if event
            else "No matching ticket events"
        )

    def showEvent(self, event):
        self.revision = None
        self.timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)
