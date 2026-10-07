"""Shared colours for the GUI.

Deliberately **not** a theme engine. There is no application-wide stylesheet and
nothing here repaints a window: every tab keeps the operating system's own look,
which is what the Live Signals, Spectrum and Model Updating tabs already had and
what the Sonification tabs are being brought back to. This module exists for the
colours that carry *meaning*, because those were inconsistent:

**The same eigenfrequency was drawn in a different colour in three tabs** —
mode 1 was cyan in Sonification, red in Spectrum and blue in Model Updating. That
is a reading error waiting to happen, so the mode colours live in one place.

Two palettes, one hue order
---------------------------
Mode 1 is blue, 2 orange, 3 green, 4 purple, everywhere. The order avoids putting
red next to green, the worst pair for colourblind readers.

* :data:`MODE_COLORS` — bright. For curves drawn on the **dark plot canvases**
  (every pyqtgraph plot in this app is dark).
* :data:`MODE_COLORS_PRINT` — saturated. For **text on native chrome** and for the
  report figures, which are printed on white.

Pick by what the colour is drawn ON, not by which tab it is in.
"""
from __future__ import annotations

#: Plot canvas background. Matches the Spectrum tab, which sets "k" directly.
PLOT_BG = "k"
#: Axis lines and ticks drawn ON the dark plot canvas.
PLOT_AXIS = "#5b6474"

# --- mode / eigenfrequency colours -----------------------------------------
#: Bright variant, for curves on a dark plot canvas. Index 0 is mode 1.
MODE_COLORS = ("#5ac8fa", "#ff9f43", "#7ee787", "#c792ea", "#45d0c8", "#f0d264")
#: Saturated variant, for text on light chrome and for the printed report
#: figures. Same hue order, so "mode 2 is the orange one" holds in both.
MODE_COLORS_PRINT = ("#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#ca8a04")


def mode_color(index: int, *, on_light: bool = False) -> str:
    """Colour for mode/eigenfrequency ``index`` (0-based), wrapping if needed.

    ``on_light`` picks the variant legible on native chrome or on paper.
    """
    palette = MODE_COLORS_PRINT if on_light else MODE_COLORS
    return palette[int(index) % len(palette)]


# --- semantic colours, for text and controls on NATIVE chrome ---------------
# Tuned to stay legible on a light background. The previous values were chosen
# against a near-black panel: #ffd93d amber in particular is unreadable on white.
#: Recording, or a destructive action.
RECORD = "#c0392b"
#: Playing, running, or a healthy state.
PLAY = "#1e8449"
#: Something the user should read: a warning or a transient status.
STATUS = "#b58900"
#: Secondary text: captions, units, hints.
DIM = "#6b7280"
#: A hairline between sections.
EDGE = "#cbd5e1"
#: The accent used for group-box titles and section headings.
ACCENT = "#2563eb"

# --- the same roles, for drawing ON a dark plot canvas ----------------------
RECORD_ON_DARK = "#ff6b6b"
PLAY_ON_DARK = "#7ee787"
STATUS_ON_DARK = "#ffd93d"
DIM_ON_DARK = "#8b93a1"


def group_css() -> str:
    """A titled section: a hairline border and an accented title, no background.

    No ``background`` rule on purpose — the box must take the window's own
    colour so the tab looks native. The Sonification tabs used to paint these
    dark, which is what made them look like a different application.
    """
    return (
        f"QGroupBox {{ border:1px solid {EDGE}; border-radius:4px;"
        f" margin-top:7px; padding:6px; }}"
        f"QGroupBox::title {{ color:{ACCENT}; font-weight:bold;"
        f" subcontrol-origin:margin; left:7px; padding:0 3px; }}"
    )


def on_light(color: str, factor: float = 0.62) -> str:
    """Darken a colour chosen for a dark canvas so it reads as text on chrome.

    The voice/role colours live in ``sonification.chorus.types``, which must not
    import from ``sensepi.gui`` (guardrail G7), and they are used both as swatches
    on dark plots and as label text on native chrome. Rather than duplicate the
    palette, the one identity is darkened here at the point it becomes text.
    """
    c = str(color).lstrip("#")
    if len(c) != 6:
        return color
    try:
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return color
    f = max(0.0, min(float(factor), 1.0))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))
