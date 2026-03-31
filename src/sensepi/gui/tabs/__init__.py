"""Tab widgets for the SensePi GUI."""

from __future__ import annotations

# TODO: move SampleKey + live buffers into a shared LiveDataStore for FFT/Signals tabs.
SampleKey = tuple[int, str]

# New shared layout signature type used by SignalsTab and FftTab.
# First tuple: sensor IDs included in the layout.
# Second tuple: channel names in the layout (e.g. "ax", "ay", "gz").
LayoutSignature = tuple[tuple[int, ...], tuple[str, ...]]
