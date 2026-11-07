"""Rich-text notes and binding reference for a block instance."""

from PySide6.QtGui import QFont, QTextCursor, QTextListFormat
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .opencv_reference import opencv_reference


class DocumentationEditor(QDialog):
    def __init__(self, block, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Documentation · {block.name}")
        self.resize(740, 520)
        self.original = block.documentation
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Document what this block does and how it is used.")
        self.editor.setHtml(self.original)
        self.editor.document().setModified(False)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        notes = QWidget()
        notes_layout = QVBoxLayout(notes)
        self.tabs.addTab(notes, "Block notes")
        toolbar = QToolBar()
        notes_layout.addWidget(toolbar)
        self.bold = toolbar.addAction("Bold")
        self.bold.setCheckable(True)
        self.bold.triggered.connect(self.set_bold)
        self.italic = toolbar.addAction("Italic")
        self.italic.setCheckable(True)
        self.italic.triggered.connect(self.editor.setFontItalic)
        self.underline = toolbar.addAction("Underline")
        self.underline.setCheckable(True)
        self.underline.triggered.connect(self.editor.setFontUnderline)
        toolbar.addAction("Bulleted list", self.add_bullets)
        self.editor.currentCharFormatChanged.connect(self.refresh_format)
        self.refresh_format(self.editor.currentCharFormat())
        notes_layout.addWidget(self.editor)

        reference = opencv_reference(block)
        if reference:
            reference_page = QWidget()
            reference_layout = QVBoxLayout(reference_page)
            browser = QTextBrowser()
            browser.setOpenExternalLinks(True)
            browser.setHtml(reference)
            reference_layout.addWidget(browser)
            copy_button = QPushButton("Copy reference to notes")
            copy_button.clicked.connect(lambda: self._copy_reference(reference))
            reference_layout.addWidget(copy_button)
            self.tabs.addTab(reference_page, "OpenCV reference")
            if not self.original.strip():
                self.tabs.setCurrentWidget(reference_page)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Save documentation")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy_reference(self, reference):
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.editor.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(reference)
        cursor.endEditBlock()
        self.editor.setTextCursor(cursor)
        self.tabs.setCurrentIndex(0)
        self.editor.setFocus()

    def set_bold(self, enabled):
        self.editor.setFontWeight(QFont.Weight.Bold if enabled else QFont.Weight.Normal)

    def add_bullets(self):
        self.editor.textCursor().createList(QTextListFormat.Style.ListDisc)
        self.editor.setFocus()

    def refresh_format(self, text_format):
        self.bold.setChecked(text_format.fontWeight() >= QFont.Weight.Bold)
        self.italic.setChecked(text_format.fontItalic())
        self.underline.setChecked(text_format.fontUnderline())

    def documentation(self):
        document = self.editor.document()
        if not document.isModified():
            return self.original
        return "" if document.isEmpty() else self.editor.toHtml()
