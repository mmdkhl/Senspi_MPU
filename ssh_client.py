from __future__ import annotations

"""
Legacy shim module.

This file exists for backward compatibility only.
New code should import from :mod:`sensepi.remote.ssh_client` instead.
"""

from sensepi.remote.ssh_client import SSHClient, SSHConfig, Host

__all__ = ["SSHClient", "SSHConfig", "Host"]
