"""Sensor-specific data models and parsers.

Each supported sensor exposes typed sample structures plus conversion helpers
that turn raw JSON lines or SSH output into those dataclasses. Currently the
:mod:`mpu6050` module defines :class:`MpuSample` used across the pipeline.
"""
