"""The one info button, and the one popover behind it.

Several tabs explain themselves with paragraphs printed straight onto the
window. That text is right, but it is permanent: it costs the same screen space
on the thousandth run as on the first, and it pushes the plots — the thing
people actually came to look at — into a corner.

This is the standard fix, **progressive disclosure**: the summary stays on
screen, the detail moves one click away. Three affordances are in play and they
are not interchangeable:

* **tooltip** — hover, transient, one short phrase on one control. The app
  already has ~60 and they stay as they are.
* **info button → popover** — click, stays until dismissed, text is selectable.
  For a paragraph, a formula, a list. That is this module.
* permanent on-screen text — for anything that must not be missed, such as a
  statement about what the measurement actually means. Those are deliberately
  *not* moved behind a click.

One implementation so every one of them looks and behaves the same.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

#: The glyph. A circled "i" reads as information in every locale and needs no
#: icon file, no theme variant and no recolouring when the palette changes.
GLYPH = "ⓘ"

#: Wide enough for a readable line, narrow enough not to cover the plot it
#: explains. Roughly 60 characters at the default size.
DEFAULT_MAX_WIDTH = 420
#: Past this the popover scrolls rather than growing off-screen.
DEFAULT_MAX_HEIGHT = 520


class _Popover(QFrame):
    """The panel the button opens.

    ``Qt.Popup`` is what makes it behave like a popover rather than a window:
    Qt grabs the mouse, so a click anywhere outside dismisses it, and Esc closes
    it, with no bookkeeping here.
    """

    def __init__(self, text: str, title: str = "", *,
                 max_width: int = DEFAULT_MAX_WIDTH,
                 max_height: int = DEFAULT_MAX_HEIGHT,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Popup)
        self._max_width = int(max_width)
        self._max_height = int(max_height)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(6)

        if title:
            heading = QLabel(title, self)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            outer.addWidget(heading)

        self._body = QLabel(self)
        self._body.setText(text)
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.RichText)
        self._body.setOpenExternalLinks(False)
        # Selectable so a frequency or a path can be copied out of the
        # explanation instead of retyped.
        self._body.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self._body.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        # Only scroll when the content is genuinely long: a two-line popover
        # with a scroll bar looks broken.
        self._scroll = QScrollArea(self)
        self._scroll.setWidget(self._body)
        # NOT widgetResizable: the label is given an explicit width below and
        # the area scrolls it. With resizing on, the label and the measurement
        # disagreed and a scroll bar appeared over two lines of text.
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setMaximumHeight(max_height)
        outer.addWidget(self._scroll)

        self.setMaximumWidth(max_width + 28)

    def set_text(self, text: str) -> None:
        self._body.setText(text)

    def _fit_body_width(self) -> None:
        """Width the text wants, capped at ``max_width``.

        A word-wrapped QLabel inside a scroll area has no useful width hint — it
        collapses to roughly half the budget, so a paragraph wrapped far
        narrower than it needed to. Measuring the rich text directly gives a
        short note a short popover and a long one the full width.
        """
        doc = QTextDocument()
        doc.setDefaultFont(self._body.font())
        doc.setHtml(self._body.text())
        doc.setTextWidth(-1)
        ideal = int(math.ceil(doc.idealWidth())) + 2
        width = max(160, min(self._max_width, ideal))

        # Size the SCROLL AREA, not the label: the area is widgetResizable, so
        # the label follows the viewport and wraps to it. Fixing the label's
        # width instead made the two fight, and the scroll bar then covered the
        # right-hand end of every line.
        # Lay the label out at that width and ask it how tall it became. This
        # is measured AFTER the fact, so it cannot disagree with what is drawn.
        self._body.setFixedWidth(width)
        self._body.adjustSize()
        needed = self._body.height()

        if needed > self._max_height:
            # It will scroll, so reserve the bar's width or it eats the text.
            bar = self._scroll.verticalScrollBar().sizeHint().width() or 15
            self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._scroll.setFixedWidth(width + bar + 4)
            self._scroll.setFixedHeight(self._max_height)
        else:
            # No bar at all: one over a two-line note just looks broken.
            self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._scroll.setFixedWidth(width + 4)
            self._scroll.setFixedHeight(needed + 4)

    def popup_at(self, anchor: QWidget) -> None:
        """Show below ``anchor``, nudged back on screen if it would overflow."""
        self._fit_body_width()
        self.adjustSize()
        pos = anchor.mapToGlobal(anchor.rect().bottomLeft())
        x, y = pos.x(), pos.y() + 4
        screen = anchor.screen()
        if screen is not None:
            area = screen.availableGeometry()
            x = min(x, area.right() - self.width() - 8)
            x = max(x, area.left() + 8)
            if y + self.height() > area.bottom():
                # No room below: flip above the button rather than clip.
                above = anchor.mapToGlobal(anchor.rect().topLeft()).y()
                y = max(area.top() + 8, above - self.height() - 4)
        self.move(x, y)
        self.show()


class InfoButton(QToolButton):
    """A small ⓘ that opens a popover explaining the section beside it.

    ``text`` is rich text. Keep it to what a reader needs *in that moment* —
    a popover nobody finishes is no better than a paragraph nobody reads.
    """

    def __init__(self, text: str, *, title: str = "",
                 max_width: int = DEFAULT_MAX_WIDTH,
                 max_height: int = DEFAULT_MAX_HEIGHT,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._title = title
        self._max_width = int(max_width)
        self._max_height = int(max_height)
        self._popover: _Popover | None = None

        self.setText(GLYPH)
        self.setAutoRaise(True)                 # flat until hovered
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setToolTip("What is this?")
        self.setAccessibleName(f"Information about {title}" if title else "Information")
        # Square and small, so it sits on a title row without changing its height.
        self.setFixedSize(18, 18)
        self.clicked.connect(self._show_popover)

    # -- content ----------------------------------------------------------
    def set_info_text(self, text: str) -> None:
        """Replace the explanation, including while the popover is open."""
        self._text = text
        if self._popover is not None:
            self._popover.set_text(text)

    def info_text(self) -> str:
        return self._text

    # -- behaviour --------------------------------------------------------
    def _show_popover(self) -> None:
        if self._popover is None:
            # Built on first use: most info buttons are never pressed, and a
            # tab can hold a dozen of them.
            self._popover = _Popover(
                self._text, self._title,
                max_width=self._max_width, max_height=self._max_height,
                parent=self)
        self._popover.popup_at(self)


def info_header(title: str, info: str, *, parent: QWidget | None = None,
                bold: bool = True, max_width: int = DEFAULT_MAX_WIDTH) -> QWidget:
    """A section heading with its info button: ``TITLE  ⓘ``.

    Returns a plain ``QWidget`` so it can go straight into any layout. The
    button is reachable as ``widget.info_button`` if the text has to change
    later.
    """
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    label = QLabel(title, row)
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    layout.addWidget(label)

    button = InfoButton(info, title=title, max_width=max_width, parent=row)
    layout.addWidget(button)
    layout.addStretch(1)

    row.label = label                   # type: ignore[attr-defined]
    row.info_button = button            # type: ignore[attr-defined]
    return row
