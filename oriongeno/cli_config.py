"""Small CLI and environment parsing helpers shared by OrionGeno commands."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("true", "1", "t", "y", "yes"):
        return True
    if normalized in ("false", "0", "f", "n", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def env_str(name, default=""):
    return os.environ.get(name, default)


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer, got {value!r}.") from exc


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return str_to_bool(value)


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        logging.error("File does not exist: %s", file_path)
        sys.exit(1)

