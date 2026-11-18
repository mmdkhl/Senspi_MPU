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
        f"QGroupBox {{ border:1px solid {edge()}; border-radius:4px;"
        f" margin-top:7px; padding:6px; }}"
        f"QGroupBox::title {{ color:{accent()}; font-weight:bold;"
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


# --- colours that must work on EITHER OS theme ------------------------------
# The chrome follows the operating system, so a fixed accent cannot be right in
# both: #2563eb reads well on white and is nearly unreadable on near-black.
# These are resolved when a widget is BUILT, which is after QApplication exists,
# so they see the palette actually in use.

def is_dark() -> bool:
    """True when the application palette is a dark one."""
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return False
    return app.palette().color(QPalette.Window).lightness() < 128


def accent() -> str:
    """Section titles and links. Lightened on dark, where saturated blue dies."""
    return "#6cb6ff" if is_dark() else "#2563eb"


def dim() -> str:
    """Secondary text."""
    return "#9aa3b2" if is_dark() else "#6b7280"


def edge() -> str:
    """Hairline borders. A light rule on a dark window is far too loud."""
    return "#3a4452" if is_dark() else "#cbd5e1"


def mode_colors() -> tuple:
    """The mode palette for TEXT on the current chrome."""
    return MODE_COLORS if is_dark() else MODE_COLORS_PRINT


def semantic(name: str) -> str:
    """RECORD / PLAY / STATUS, picked for the current chrome."""
    pair = {
        "record": (RECORD_ON_DARK, RECORD),
        "play": (PLAY_ON_DARK, PLAY),
        "status": (STATUS_ON_DARK, STATUS),
    }[name]
    return pair[0] if is_dark() else pair[1]


def role_text(color: str) -> str:
    """A voice/role colour, made legible as TEXT on the current chrome.

    The role colours were picked against a dark canvas, so on a light window
    they are washed out and have to be darkened; on a dark window they are
    already right and darkening them would be exactly wrong.
    """
    return color if is_dark() else on_light(color)


# --- embedded matplotlib ----------------------------------------------------
# matplotlib does not read Qt stylesheets or the Qt palette, so an embedded
# canvas stays white in a dark window. These follow the palette like everything
# else. Figures that are SAVED for a report are not passed through here: those
# are printed and embedded in papers, so they keep matplotlib's white.

def canvas_bg() -> str:
    """The surface a figure is drawn on.

    Black on dark, to match PLOT_BG: every pyqtgraph plot in the app draws on
    black, so a lighter matplotlib canvas beside one reads as a different
    application.
    """
    return "#000000" if is_dark() else "#F7F9FC"


def canvas_fg() -> str:
    return "#c7ced9" if is_dark() else "#1a1f27"


def model_line() -> str:
    """A structural member. Neutral, so it is never read as a mode colour."""
    return "#9fb4cc" if is_dark() else "#123B6D"


def model_reference() -> str:
    """The undeformed reference, and floors with no sensor."""
    return "#4a5666" if is_dark() else "#B9C2CC"


def style_mpl_canvas(figure, *, axes=None) -> None:
    """Make an on-screen matplotlib figure match the current chrome.

    Safe to call again after an ``ax.clear()``, which drops the styling.
    """
    from matplotlib.colors import to_rgba

    if figure is None:
        return
    bg, fg = canvas_bg(), canvas_fg()
    figure.set_facecolor(bg)
    for ax in (axes if axes is not None else figure.get_axes()):
        if ax is None:
            continue
        try:
            ax.set_facecolor(bg)
            for spine in ax.spines.values():
                spine.set_color(fg)
            ax.tick_params(colors=fg, which="both")
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            if ax.get_title():
                ax.title.set_color(fg)
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(bg)
                legend.get_frame().set_edgecolor(fg)
                for text in legend.get_texts():
                    text.set_color(fg)
            # 3D axes carry a third axis and axis LINES that default to black
            # and are not covered by the 2D spine path.
            zaxis = getattr(ax, "zaxis", None)
            if zaxis is not None:
                zaxis.label.set_color(fg)
                for pane_axis in (ax.xaxis, ax.yaxis, zaxis):
                    try:
                        pane_axis.line.set_color(fg)
                    except Exception:
                        pass
                    # The panes are the three background walls of the box. They
                    # default to a light grey, which on a dark canvas reads as a
                    # solid block rather than as a frame around the structure.
                    try:
                        pane_axis.pane.set_facecolor(bg)
                        pane_axis.pane.set_edgecolor(bg)
                        pane_axis.pane.set_alpha(1.0)
                    except Exception:
                        try:
                            pane_axis.set_pane_color(to_rgba(bg))
                        except Exception:
                            pass
        except Exception:      # never let styling break a redraw
            continue
