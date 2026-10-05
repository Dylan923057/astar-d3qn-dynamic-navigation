from __future__ import annotations

from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path, _seen: set[Path] | None = None
) -> dict[str, Any]:
    import yaml

    resolved = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if resolved in seen:
        raise ValueError(f"Circular config inheritance involving: {resolved}")
    seen.add(resolved)
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    inherited = data.pop("inherits", None)
    if inherited is None:
        return data
    base_path = Path(inherited)
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return _deep_merge(load_config(base_path, seen), data)


def deep_get(data: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current

