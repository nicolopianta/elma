"""
A small "?" button that shows explanatory text in a dismissable popup next
to it when clicked, instead of that text sitting permanently on the form.

Built on Qt's own Qt.Popup window type, which already closes itself on any
click outside it (or on losing focus) -- no extra event wiring needed for
the "click elsewhere dismisses it" behavior the caller gets for free.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class HelpButton(QToolButton):
    """A circular "?" button; click to show `text` in a popup, click
    anywhere else to dismiss it."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText("?")
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "QToolButton {"
            "  border-radius: 9px; border: 1px solid palette(mid);"
            "  background: palette(button); font-weight: bold; font-size: 11px;"
            "  padding: 0px; }"
            "QToolButton:hover { background: palette(light); }"
        )
        self.setToolTip("Click for details")
        self._text = text
        self.clicked.connect(self._show_popup)

    def _show_popup(self):
        popup = QWidget(self, Qt.Popup)
        popup.setAttribute(Qt.WA_DeleteOnClose)
        popup.setStyleSheet(
            "QWidget { background-color: palette(window); border: 1px solid palette(mid); }"
        )
        layout = QVBoxLayout(popup)
        layout.setContentsMargins(8, 6, 8, 6)
        label = QLabel(self._text)
        label.setWordWrap(True)
        label.setMaximumWidth(340)
        layout.addWidget(label)
        popup.move(self.mapToGlobal(self.rect().bottomLeft()))
        popup.show()


def help_row(help_text: str, caption: str | None = None) -> QWidget:
    """
    A small horizontal row holding just a HelpButton (left-aligned),
    optionally preceded by a short gray caption -- for a group-level
    explanation that isn't tied to one specific field, dropped in as its
    own row (e.g. layout.addWidget(help_row(...))) rather than attached
    next to a field.
    """
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    if caption:
        label = QLabel(caption)
        label.setStyleSheet("color: gray;")
        h.addWidget(label)
    h.addWidget(HelpButton(help_text))
    h.addStretch(1)
    return row


def with_help(widget: QWidget, help_text: str) -> QWidget:
    """
    Wraps `widget` together with a HelpButton in a horizontal row, for
    attaching a "?" directly next to a single form field
    (form.addRow(label, with_help(spin_box, "..."))).
    """
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    widget.setSizePolicy(QSizePolicy.Expanding, widget.sizePolicy().verticalPolicy())
    h.addWidget(widget)
    h.addWidget(HelpButton(help_text))
    return row
