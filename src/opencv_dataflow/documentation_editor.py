"""Rich-text notes for a block instance."""

from PySide6.QtGui import QFont, QTextListFormat
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit, QToolBar, QVBoxLayout


class DocumentationEditor(QDialog):
    def __init__(self, block, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Documentation · {block.name}")
        self.resize(640, 420)
        self.original = block.documentation
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Document what this block does and how it is used.")
        self.editor.setHtml(self.original)
        self.editor.document().setModified(False)

        layout = QVBoxLayout(self)
        toolbar = QToolBar()
        layout.addWidget(toolbar)
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
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Save documentation")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
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
