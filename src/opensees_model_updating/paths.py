"""Where this package writes its output.

Every writer in this package used to build a bare relative path — ``"output/…"``
— in fourteen places, including four OpenSees ``recorder`` calls whose filenames
cross into the C++ layer. A bare relative path resolves against the **current
working directory**, which was fine when this was a standalone tool run from its
own folder, and stopped being fine when it was embedded in an application that
runs several threads at once.

The host application worked around it with ``os.chdir()``. That is a
**process-global** call: it changes the working directory for every thread, not
just the caller. While a calibration ran, a live acquisition thread resolving a
relative path silently got a different directory. Nothing depended on that in
practice, but the failure it invites — a file written somewhere nobody looks —
is invisible and expensive to chase.

This module removes the need for the workaround while **preserving standalone
behaviour exactly**: with no argument and no environment variable,
:func:`output_dir` returns ``Path("output")``, resolved against the working
directory, which is what every caller in this package did before. A host that
wants somewhere specific passes it in. Nobody running this package on its own has
to change anything.

Resolution order
----------------
1. an explicit ``base`` argument — what the embedding application passes;
2. ``$OPENSEES_OUTPUT_DIR`` — for scripted runs and CI;
3. ``Path("output")`` — the historical default, unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "OPENSEES_OUTPUT_DIR"
DEFAULT_NAME = "output"


def output_dir(base=None, *, create: bool = True) -> Path:
    """The directory this package should write to.

    ``create`` is on by default because every caller here writes immediately
    after asking; pass ``False`` when only the location is wanted.
    """
    if base is not None:
        path = Path(base)
    else:
        env = os.environ.get(ENV_VAR)
        path = Path(env).expanduser() if env else Path(DEFAULT_NAME)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(name: str, base=None, *, create: bool = True) -> Path:
    """A file inside the output directory.

    Returned as a ``Path``; callers that must hand a *string* to OpenSees should
    use :func:`output_str`, which is explicit about why.
    """
    return output_dir(base, create=create) / str(name)


def output_str(name: str, base=None, *, create: bool = True) -> str:
    """A file path as a string, for the OpenSees ``recorder`` API.

    ``ops.recorder`` takes a filename that crosses into the C++ layer, where a
    relative path is resolved against the process working directory and a wrong
    one produces no error at all — the file is simply written somewhere else.
    Passing an absolute path is what makes that impossible.
    """
    return str(output_path(name, base, create=create))
