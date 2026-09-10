from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from devbuilder.paths import paths

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
_PLACEHOLDER_PATTERN = re.compile(r"\$\{[^}]*\}")


def _validate_placeholders(value: str) -> None:
    """
    Reject malformed environment placeholders.

    Supported forms are ${VAR} and ${VAR:-default}.
    """
    for placeholder in _PLACEHOLDER_PATTERN.findall(value):
        if _ENV_PATTERN.fullmatch(placeholder) is None:
            raise ValueError(f"Invalid environment placeholder: {placeholder}")

    remainder = _PLACEHOLDER_PATTERN.sub("", value)
    if "${" in remainder:
        raise ValueError(f"Unclosed environment placeholder in: {value!r}")


def _resolve_env(match: re.Match[str]) -> str:
    """
    Resolve one ${VAR} or ${VAR:-default} expression.

    Empty values are treated as unset. A missing or empty variable without
    a default raises ValueError.
    """
    variable = match.group(1)
    default = match.group(2)
    value = os.environ.get(variable)

    if value:
        return value

    if default is not None:
        return default

    raise ValueError(f"Required environment variable is unset or empty: {variable}")


def _parse_json_scalar(value: str) -> Any:
    """
    Parse conservative JSON scalar syntax, preserving other strings unchanged.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value

    if parsed is None or type(parsed) in (str, int, float, bool):
        return parsed

    return value


def _expand(value: Any) -> Any:
    """
    Recursively expand environment placeholders in values.

    Dictionary keys are not expanded.
    """
    if isinstance(value, str):
        _validate_placeholders(value)

        match = _ENV_PATTERN.fullmatch(value)
        if match is not None:
            return _parse_json_scalar(_resolve_env(match))

        return _ENV_PATTERN.sub(_resolve_env, value)

    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_expand(item) for item in value]

    return value


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load a YAML configuration file and expand environment placeholders.
    """
    path = Path(config_path) if config_path is not None else paths.configs / "settings.yaml"

    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        loaded = {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    expanded = _expand(loaded)
    logger.info("Loaded config from %s", path)
    return expanded
