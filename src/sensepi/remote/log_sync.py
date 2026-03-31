"""Helpers for syncing log files from a Raspberry Pi over SFTP."""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sensepi.config.app_config import AppPaths, normalize_remote_path
from sensepi.config.log_paths import (
    LOG_SUBDIR_MPU,
    build_pc_session_root,
    slugify_session_name,
)
from sensepi.remote.ssh_client import Host, SSHClient


@dataclass
class SyncReport:
    """Outcome of a log sync operation."""

    remote_root: str
    local_root: Path
    downloaded: list[Path]
    skipped: int


def _iter_remote_files(sftp, root: str):
    """Yield ``(path, size, mtime)`` for files under *root* (depth-first)."""

    stack = [root]
    while stack:
        cur = stack.pop()
        for attr in sftp.listdir_attr(cur):
            rp = f"{cur.rstrip('/')}/{attr.filename}"
            if stat.S_ISDIR(attr.st_mode):
                stack.append(rp)
            else:
                yield rp, int(attr.st_size), int(attr.st_mtime)


def _should_sync(remote_size: int, remote_mtime: int, local_path: Path) -> bool:
    """Return True if the remote file should be downloaded."""

    if not local_path.exists():
        return True
    try:
        st = local_path.stat()
        if st.st_size != remote_size:
            return True
        return int(st.st_mtime) < remote_mtime
    except OSError:
        return True


def _candidate_roots(host_cfg, session_slug: str | None) -> list[str]:
    """Return ordered remote directory candidates for log discovery."""

    data_root = normalize_remote_path(str(host_cfg.data_dir), user=host_cfg.user)
    base_root = normalize_remote_path(str(host_cfg.base_path), user=host_cfg.user)

    candidates = [
        str(PurePosixPath(data_root) / LOG_SUBDIR_MPU),
        str(PurePosixPath(base_root) / "logs"),
        str(PurePosixPath(f"/home/{host_cfg.user}/logs") / LOG_SUBDIR_MPU),
    ]

    search_order: list[str] = []
    if session_slug:
        search_order.extend(str(PurePosixPath(c) / session_slug) for c in candidates)
    search_order.extend(candidates)

    seen: set[str] = set()
    unique_paths: list[str] = []
    for path in search_order:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def sync_logs_from_pi(
    host_cfg,
    session_name: str | None,
    *,
    raw_root: Path | None = None,
) -> SyncReport:
    """Download new log files from a Raspberry Pi into the local raw data tree."""

    session_name = (session_name or "").strip() or None
    session_slug = slugify_session_name(session_name) if session_name else None

    raw_root = raw_root or AppPaths().raw_data

    local_root = build_pc_session_root(
        raw_root=raw_root,
        host_slug=host_cfg.name,
        session_name=session_name,
        sensor_prefix=LOG_SUBDIR_MPU,
    )
    local_root.mkdir(parents=True, exist_ok=True)

    remote_host = Host(
        name=host_cfg.name,
        host=host_cfg.host,
        user=host_cfg.user,
        password=host_cfg.password,
        port=host_cfg.port,
    )

    client = SSHClient(remote_host)
    try:
        candidates = _candidate_roots(host_cfg, session_slug)
        remote_root = None
        for candidate in candidates:
            if client.path_exists(candidate):
                remote_root = candidate
                break

        if remote_root is None:
            raise FileNotFoundError(
                f"No remote log directory found (tried: {candidates})"
            )

        downloaded: list[Path] = []
        skipped = 0

        exts = {".csv", ".jsonl"}
        meta_suffix = ".meta.json"

        with client.sftp() as sftp:
            for rp, rsize, rmtime in _iter_remote_files(sftp, remote_root):
                name = PurePosixPath(rp).name
                if not (name.endswith(meta_suffix) or PurePosixPath(rp).suffix in exts):
                    continue

                rel = PurePosixPath(rp).relative_to(PurePosixPath(remote_root))
                lp = local_root / Path(rel.as_posix())
                lp.parent.mkdir(parents=True, exist_ok=True)

                if _should_sync(rsize, rmtime, lp):
                    sftp.get(rp, str(lp))
                    try:
                        os.utime(lp, (time.time(), rmtime))
                    except OSError:
                        pass
                    downloaded.append(lp)
                else:
                    skipped += 1

        return SyncReport(
            remote_root=remote_root,
            local_root=local_root,
            downloaded=downloaded,
            skipped=skipped,
        )
    finally:
        client.close()
