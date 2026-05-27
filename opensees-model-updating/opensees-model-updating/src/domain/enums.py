# -*- coding: utf-8 -*-
"""Domain enumerations."""

from enum import Enum


class ColumnOrientation(str, Enum):
    WEAK = "weak"
    STRONG = "strong"


class AnalysisType(str, Enum):
    MODAL = "modal"
    GRAVITY = "gravity"
    TRANSIENT = "transient"


class MassCalibrationScope(str, Enum):
    SELF_WEIGHT_ONLY = "self_weight_only"
    TOTAL_MASS = "total_mass"
