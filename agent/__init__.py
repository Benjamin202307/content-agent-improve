"""Shared runtime safeguards for every Content Agent entry point."""

import sys


def _configure_console_output() -> None:
    """Prevent non-GBK diagnostic symbols from terminating generation on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


_configure_console_output()
