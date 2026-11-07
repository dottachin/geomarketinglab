"""Structured-enough logging (doc §9: structured logging). One line per event, key=value."""

from __future__ import annotations

import logging
import sys

from geomlab.settings import get_settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
        )
        root = logging.getLogger("geomlab")
        root.addHandler(handler)
        root.setLevel(get_settings().log_level.upper())
        root.propagate = False
        _CONFIGURED = True
    if not name.startswith("geomlab"):
        name = f"geomlab.{name}"
    return logging.getLogger(name)
