from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BaselineState:
    """Simple per-channel baseline offset."""

    offset: Optional[np.ndarray] = None
    active: bool = False

    def apply(self, sample: np.ndarray) -> np.ndarray:
        """Apply baseline if available, otherwise return sample unchanged."""

        if self.active and self.offset is not None:
            return sample - self.offset
        return sample


def collect_baseline_samples(samples: List[np.ndarray]) -> np.ndarray:
    """
    Stack a list of samples and compute mean per channel.

    ``samples``: list of arrays shaped ``(n_channels,)`` or ``(n_axes,)``.
    """
    if not samples:
        raise ValueError("collect_baseline_samples() requires at least one sample")

    stacked = np.stack(samples, axis=0)  # shape: (N, n_channels)
    return stacked.mean(axis=0)
