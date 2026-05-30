"""
opensees_model_updating
=======================

Automated OpenSees modal calibration and model updating for 3-D aluminum frames.

Submodules import openseespy at function call time, not at package load time,
so this package can be imported safely even when openseespy is not installed.
"""

__version__ = "1.0.0"
