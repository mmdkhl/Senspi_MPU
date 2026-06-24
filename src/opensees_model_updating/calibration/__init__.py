# -*- coding: utf-8 -*-
"""Calibration package.

The deterministic calibrator symbols (``run_calibration``, ``modal_residuals``,
``apply_calibration_vector``, ``prepare_experimental_modal_data``) are resolved
**lazily** via :pep:`562` ``__getattr__`` so that importing this package — and its
pure submodules such as :mod:`bayesian` — does NOT pull in openseespy at load time.
The forward-model import only happens the moment a calibrator symbol is actually
accessed. This matches the package contract stated in
``opensees_model_updating/__init__.py`` ("submodules import openseespy at function
call time, not at package load time"), and lets the pure Bayesian math and the
partial-coverage residual tests run without the OpenSees extra installed.
"""

__all__ = [
    "prepare_experimental_modal_data",
    "apply_calibration_vector",
    "modal_residuals",
    "run_calibration",
]


def __getattr__(name):
    # Lazy re-export of the deterministic calibrator API (needs openseespy).
    if name in __all__:
        from . import calibrator
        return getattr(calibrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
