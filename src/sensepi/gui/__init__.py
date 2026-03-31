"""Desktop GUI implementation built with PySide6/Qt.

Panels in :mod:`gui.tabs` surface configuration, live FFT/streaming plots, and
offline log viewers, while :mod:`gui.widgets` houses shared Qt components.
This layer orchestrates the Qt event loop and delegates streaming/recording to
the :mod:`sensepi.remote` and :mod:`sensepi.core` packages.
"""
