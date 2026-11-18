"""The shared info button: one implementation, used by every tab.

Several tabs printed paragraphs permanently onto the window, which costs the
same screen space on the thousandth run as on the first and pushes the plots
aside. The explanation moves into a popover; the summary stays on screen.

What matters here is that it behaves the same everywhere: opens on click,
dismisses on click-outside or Esc without leaving a stray window, stays on
screen near an edge, and does not change the height of the row it sits in.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication, QLabel, QWidget


class TestInfoButton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def _button(self, text="Explains the thing.", **kw):
        from sensepi.gui.widgets.info_button import InfoButton

        b = InfoButton(text, **kw)
        self.addCleanup(b.deleteLater)
        return b

    def test_it_is_small_and_flat_so_it_does_not_resize_its_row(self):
        b = self._button()

        self.assertEqual(b.size().width(), 18)
        self.assertEqual(b.size().height(), 18)
        self.assertTrue(b.autoRaise(), "the button should be flat until hovered")

    def test_it_carries_a_hover_hint_as_well(self):
        # The info button is itself discoverable: hovering says what it does,
        # which is the behaviour the rest of the app already uses.
        b = self._button(title="Spectrum")

        self.assertTrue(b.toolTip())
        self.assertIn("Spectrum", b.accessibleName())

    def test_no_popover_is_built_until_it_is_pressed(self):
        # A tab can hold a dozen of these and most are never opened.
        b = self._button()

        self.assertIsNone(b._popover)

    def test_clicking_opens_a_popover_with_the_text(self):
        b = self._button("<b>Mode 1</b> is the first bending mode.")
        self.addCleanup(lambda: b._popover and b._popover.close())

        b.click()

        self.assertIsNotNone(b._popover)
        self.assertTrue(b._popover.isVisible())
        self.assertIn("first bending mode", b._popover._body.text())

    def test_the_popover_is_a_popup_so_it_dismisses_itself(self):
        # Qt.Popup is what makes a click anywhere outside, and Esc, close it
        # without any bookkeeping of our own.
        b = self._button()
        self.addCleanup(lambda: b._popover and b._popover.close())
        b.click()

        self.assertTrue(b._popover.windowFlags() & Qt.Popup)

    def test_the_popover_is_reused_rather_than_stacking_up(self):
        b = self._button()
        self.addCleanup(lambda: b._popover and b._popover.close())

        b.click()
        first = b._popover
        b._popover.close()
        b.click()

        self.assertIs(b._popover, first, "a second press built a second popover")

    def test_the_text_is_selectable(self):
        # So a frequency or a path can be copied out instead of retyped.
        b = self._button("f1 = 2.31 Hz")
        self.addCleanup(lambda: b._popover and b._popover.close())
        b.click()

        flags = b._popover._body.textInteractionFlags()
        self.assertTrue(flags & Qt.TextSelectableByMouse)

    def test_long_text_scrolls_instead_of_growing_off_screen(self):
        from sensepi.gui.widgets.info_button import DEFAULT_MAX_HEIGHT

        b = self._button("<br>".join(f"line {i}" for i in range(300)))
        self.addCleanup(lambda: b._popover and b._popover.close())

        b.click()

        self.assertLessEqual(b._popover._scroll.maximumHeight(), DEFAULT_MAX_HEIGHT)
        self.assertLessEqual(b._popover.width(), b._popover.maximumWidth())

    def test_a_short_note_gets_no_scroll_bar(self):
        # A scroll bar over two lines of text looks broken, and an early version
        # showed one on every popover.
        b = self._button("Where the energy is.")
        self.addCleanup(lambda: b._popover and b._popover.close())

        b.click()

        self.assertFalse(b._popover._scroll.verticalScrollBar().isVisible())

    def test_a_paragraph_is_not_clipped_by_the_scroll_bar(self):
        # The label and the scroll area disagreed about width, so the bar sat on
        # top of the right-hand end of every line.
        para = ("Each mode sings as the animal type you chose, and inside that type "
                "the mode's frequency picks the species. If the structure softens, "
                "its frequencies drop and the lead moves to a lower-pitched species.")
        b = self._button(para)
        self.addCleanup(lambda: b._popover and b._popover.close())

        b.click()
        pop = b._popover

        self.assertFalse(pop._scroll.verticalScrollBar().isVisible(),
                         "a paragraph that fits was given a scroll bar")
        # The whole label must be inside the viewport, not cut off by it.
        self.assertLessEqual(pop._body.width(), pop._scroll.viewport().width())
        self.assertLessEqual(pop._body.height(), pop._scroll.viewport().height())

    def test_a_short_note_gets_a_smaller_popover_than_a_long_one(self):
        short = self._button("Short.")
        long_ = self._button("word " * 400)
        self.addCleanup(lambda: short._popover and short._popover.close())
        self.addCleanup(lambda: long_._popover and long_._popover.close())

        short.click()
        long_.click()

        self.assertLess(short._popover.height(), long_._popover.height())

    def test_the_text_can_be_replaced_later(self):
        b = self._button("before")
        self.addCleanup(lambda: b._popover and b._popover.close())
        b.click()

        b.set_info_text("after")

        self.assertEqual(b.info_text(), "after")
        self.assertIn("after", b._popover._body.text())

    def test_it_stays_on_screen_when_opened_at_the_right_edge(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        host.resize(300, 80)
        b = self._button(parent=host)
        screen = self.app.primaryScreen()
        if screen is None:
            self.skipTest("no screen")
        area = screen.availableGeometry()
        host.move(area.right() - 40, area.top() + 10)
        host.show()
        self.addCleanup(lambda: b._popover and b._popover.close())

        b.click()

        self.assertLessEqual(b._popover.x() + b._popover.width(), area.right())
        self.assertGreaterEqual(b._popover.x(), area.left())


class TestInfoHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def test_it_pairs_a_heading_with_its_button(self):
        from sensepi.gui.widgets.info_button import InfoButton, info_header

        row = info_header("RESONANCE RADAR", "Where the energy is.")
        self.addCleanup(row.deleteLater)

        self.assertIsInstance(row.info_button, InfoButton)
        self.assertIsInstance(row.label, QLabel)
        self.assertEqual(row.label.text(), "RESONANCE RADAR")
        self.assertTrue(row.label.font().bold())
        self.assertEqual(row.info_button.info_text(), "Where the energy is.")


if __name__ == "__main__":
    unittest.main()
