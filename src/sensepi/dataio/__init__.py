"""Data input/output helpers (CSV/JSON logs and file paths).

Utility modules here keep disk-level concerns isolated from the rest of the
application:
- :mod:`csv_writer` emits structured sensor logs.
- :mod:`log_loader` parses CSV/JSON recordings for offline review.
- :mod:`file_paths` centralises directory layout for logs and cache files.
"""
