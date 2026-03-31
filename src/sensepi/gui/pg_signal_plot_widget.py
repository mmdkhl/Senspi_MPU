"""Compatibility shim for the PyQtGraph SignalPlotWidget implementation.

New code should import :class:`SignalPlotWidget` from ``tab_signals.py`` (or a
future widgets module). This alias remains solely to keep the historic
``pg_signal_plot_widget`` import path alive for older code.
"""

from __future__ import annotations

from .tabs.tab_signals import SignalPlotWidget

# Historically, the PyQtGraph implementation lived in this module. Keep the
# import path alive so older code can still request PyQtGraphSignalPlotWidget.
PyQtGraphSignalPlotWidget = SignalPlotWidget

__all__ = [
    "SignalPlotWidget",
    "PyQtGraphSignalPlotWidget",
]

