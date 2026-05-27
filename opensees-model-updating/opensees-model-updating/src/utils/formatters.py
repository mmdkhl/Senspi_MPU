# -*- coding: utf-8 -*-
"""Numeric formatting utilities."""

import numpy as np

ROUND_DECIMALS = 3


def r3(x):
    """Round a value to 3 decimal places."""
    try:
        return round(float(x), ROUND_DECIMALS)
    except Exception:
        return x


def sci3(x):
    """Format a value in scientific notation with 3 decimal places."""
    try:
        return f"{float(x):.3e}"
    except Exception:
        return str(x)


def deep_round(obj, decimals=ROUND_DECIMALS):
    """Recursively round all numeric values in a nested structure."""
    if isinstance(obj, dict):
        return {k: deep_round(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_round(v, decimals) for v in obj]
    if isinstance(obj, tuple):
        return [deep_round(v, decimals) for v in obj]
    if isinstance(obj, np.ndarray):
        return [deep_round(v, decimals) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), decimals)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def story_column_layout_to_text(layout, nStory):
    """Format column layout dict as human-readable text."""
    parts = []
    for story in range(1, nStory + 1):
        cols = layout.get(story, [1, 2, 3, 4])
        parts.append(f"{story}:{','.join(str(c) for c in cols)}")
    return "; ".join(parts)


def column_orientation_layout_to_text(layout, nStory):
    """Format column orientation dict as human-readable text."""
    parts = []
    for story in range(1, nStory + 1):
        col_text = []
        for c in (1, 2, 3, 4):
            col_text.append(f"{c}={layout.get(story, {}).get(c, 'weak')}")
        parts.append(f"{story}:{','.join(col_text)}")
    return "; ".join(parts)


def additional_masses_to_text(layout, nStory):
    """Format additional mass placement dict as human-readable text."""
    parts = []
    for story in range(1, nStory + 1):
        vals = layout.get(story, [0.0, 0.0, 0.0, 0.0, 0.0])
        parts.append(f"{story}:{','.join(str(r3(v)) for v in vals)}")
    return "; ".join(parts)
