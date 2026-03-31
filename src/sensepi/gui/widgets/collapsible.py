from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """
    Simple collapsible section: a header with an arrow and a content area.
    Use setContentLayout() to put your real widgets inside.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        collapsed: bool = False,
    ) -> None:
        super().__init__(parent)

        self._toggle = QToolButton(text=title, checkable=True, checked=not collapsed)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._toggle.toggled.connect(self._on_toggled)

        self._content = QWidget()
        self._content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._content.setVisible(not collapsed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

    def setContentLayout(self, layout: QLayout) -> None:
        """
        Put a layout with your controls inside the collapsible content area.

        Raises:
            RuntimeError: if a layout is already set.
        """
        if self._content.layout() is not None:
            raise RuntimeError("Content layout already set on CollapsibleSection")
        self._content.setLayout(layout)

    def setCollapsed(self, collapsed: bool) -> None:
        """Programmatic collapse/expand (e.g. when Start is pressed)."""
        self._toggle.setChecked(not collapsed)

    def isCollapsed(self) -> bool:
        """Return True when the section is collapsed."""
        return not self._toggle.isChecked()

    def isExpanded(self) -> bool:
        """Return True when the section is expanded."""
        return self._toggle.isChecked()

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
