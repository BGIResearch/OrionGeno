"""Small CLI and environment parsing helpers shared by OrionGeno commands."""

from __future__ import annotations

import argparse
import logging
import os
import sys

AUTO_VALUE = "auto"
AUTO_VALUES = {AUTO_VALUE}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("true", "1", "t", "y", "yes"):
        return True
    if normalized in ("false", "0", "f", "n", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def is_auto(value):
    return isinstance(value, str) and value.strip().lower() in AUTO_VALUES


def _parse_auto_int(value, *, minimum):
    if is_auto(value):
        return AUTO_VALUE
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        qualifier = "positive" if minimum > 0 else "non-negative"
        raise argparse.ArgumentTypeError(
            f"Expected {qualifier} integer or 'auto', got {value!r}."
        ) from exc
    if parsed < minimum:
        qualifier = "positive" if minimum > 0 else "non-negative"
        raise argparse.ArgumentTypeError(
            f"Expected {qualifier} integer or 'auto', got {value!r}."
        )
    return parsed


def auto_positive_int(value):
    return _parse_auto_int(value, minimum=1)


def auto_nonnegative_int(value):
    return _parse_auto_int(value, minimum=0)


def env_str(name, default=""):
    return os.environ.get(name, default)


def env_auto_positive_int(name, default=AUTO_VALUE):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return auto_positive_int(value)


def env_auto_nonnegative_int(name, default=AUTO_VALUE):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return auto_nonnegative_int(value)


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
