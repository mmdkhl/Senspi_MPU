"""Data models for the Digital Twin analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DigitalTwinAnalysisParams:
    """User-editable inputs for the OpenSees digital twin model."""

    gm_file: Path
    output_dir: Path
    story_heights: list[float] = field(default_factory=lambda: [0.24, 0.24, 0.24])
    floor_masses: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.3])
    lx_m: float = 0.245
    ly_m: float = 0.23
    youngs_modulus_pa: float = 200e9
    poissons_ratio: float = 0.33
    column_thickness_m: float = 0.001
    column_width_m: float = 0.006
    beam_width_m: float = 0.001
    beam_height_m: float = 0.008
    num_modes: int = 4
    damping_ratio: float = 0.005
    dt_seconds: float = 0.01
    gm_factor: float = 9.81

    def validate(self) -> None:
        if not self.gm_file.exists():
            raise FileNotFoundError(f"Ground-motion file not found: {self.gm_file}")
        if not self.story_heights:
            raise ValueError("At least one story height is required.")
        if len(self.story_heights) != len(self.floor_masses):
            raise ValueError("story_heights and floor_masses must have the same length.")
        if self.num_modes < 1:
            raise ValueError("num_modes must be at least 1.")
        if self.dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive.")
        if self.damping_ratio < 0.0:
            raise ValueError("damping_ratio must be non-negative.")
        if any(h <= 0.0 for h in self.story_heights):
            raise ValueError("All story heights must be positive.")
        if any(m <= 0.0 for m in self.floor_masses):
            raise ValueError("All floor masses must be positive.")


@dataclass
class DigitalTwinModalResult:
    periods_s: list[float]
    frequencies_hz: list[float]
    lambdas: list[float]
    master_nodes: list[int]
    mode_shapes: dict[int, dict[int, tuple[float, float, float, float, float, float]]]


@dataclass
class DigitalTwinTransientResult:
    time_s: list[float]
    roof_disp_x_m: list[float]
    roof_accel_x_ms2: list[float]
    floor_disp_x_m: dict[int, list[float]]
    floor_accel_x_ms2: dict[int, list[float]]


@dataclass
class DigitalTwinAnalysisResult:
    params: DigitalTwinAnalysisParams
    modal: DigitalTwinModalResult
    transient: DigitalTwinTransientResult
    output_dir: Path
