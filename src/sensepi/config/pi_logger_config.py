"""Helpers for configuring the Raspberry Pi logger."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping
import shlex

from .sampling import SamplingConfig


@dataclass
class PiLoggerConfig:
    """
    Concrete configuration for the Pi logger derived from SamplingConfig.
    """

    device_rate_hz: float
    record_decimate: int = 1
    stream_decimate: int = 1
    record_rate_hz: float = 0.0
    stream_rate_hz: float = 0.0
    sections: Dict[str, Any] | None = None
    logger_script: str = "mpu6050_multi_logger.py"
    extra_cli: Mapping[str, Any] | None = None

    @classmethod
    def from_sampling(cls, sampling: SamplingConfig, **kwargs: Any) -> "PiLoggerConfig":
        """
        Construct a PiLoggerConfig from the SamplingConfig source of truth.
        """

        rate = float(sampling.device_rate_hz)
        return cls(
            device_rate_hz=rate,
            record_decimate=1,
            stream_decimate=1,
            record_rate_hz=rate,
            stream_rate_hz=rate,
            **kwargs,
        )

    # ------------------------------------------------------------------ serialization
    def to_pi_config_dict(self) -> Dict[str, Any]:
        """
        Convert into a mapping suitable for YAML serialization.
        """

        rate = float(self.device_rate_hz)
        data: Dict[str, Any] = {
            # Single-rate invariant: device/record/stream all share the same rate and decimation=1
            "device_rate_hz": rate,
            "record_decimate": 1,
            "stream_decimate": 1,
        }
        extra_sections = self.sections or {}
        for key, value in extra_sections.items():
            data[key] = value
        return data

    def render_pi_config_yaml(self) -> str:
        """
        Return the generated pi_config.yaml text with the DO NOT EDIT header.
        """

        from textwrap import dedent

        import yaml

        header = dedent(
            """
            # GENERATED FILE - DO NOT EDIT BY HAND
            # This file is derived from SamplingConfig and PiLoggerConfig.
            """
        ).strip("\n")
        body = yaml.safe_dump(self.to_pi_config_dict(), sort_keys=False)
        return header + "\n\n" + body

    def write_pi_config_yaml(self, path: Path) -> None:
        """
        Write the generated configuration to *path*.
        """

        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_pi_config_yaml())

    def build_command(self, extra_cli: str | None = None) -> List[str]:
        """Construct the logger command including any additional CLI flags."""

        cmd = build_logger_command(self)
        extra = extra_cli.strip() if isinstance(extra_cli, str) else ""
        if extra:
            cmd.extend(shlex.split(extra))
        return cmd


def _format_extra_flags(extra: Mapping[str, Any] | None) -> List[str]:
    if not extra:
        return []

    flags: List[str] = []
    for key, value in extra.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif value is not None:
            flags.extend([flag, str(value)])
    return flags


def build_logger_args(pi_cfg: PiLoggerConfig) -> List[str]:
    """Build argument list for ``mpu6050_multi_logger.py`` using ``pi_cfg``."""

    args: List[str] = [
        "--sample-rate-hz",
        f"{pi_cfg.device_rate_hz:.0f}",
    ]

    args.extend(_format_extra_flags(pi_cfg.extra_cli))
    return args


def build_logger_command(pi_cfg: PiLoggerConfig) -> List[str]:
    """
    Construct the command used to start the Pi logger process.
    """

    parts: List[str] = ["python3", "-u", pi_cfg.logger_script]
    parts.extend(build_logger_args(pi_cfg))
    return parts
