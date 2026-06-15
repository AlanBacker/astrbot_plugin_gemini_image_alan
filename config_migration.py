from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _merge_config(defaults: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    merged = defaults.copy()
    for key, value in legacy.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def migrate_legacy_config(
    config: Any,
    *,
    current_plugin_name: str,
    legacy_plugin_name: str,
) -> bool:
    """Migrate the old plugin config when the renamed fork is first installed."""
    config_path_value = getattr(config, "config_path", "")
    if not config_path_value or not getattr(config, "first_deploy", False):
        return False

    config_path = Path(config_path_value)
    expected_name = f"{current_plugin_name}_config.json"
    if config_path.name != expected_name:
        return False

    legacy_path = config_path.with_name(f"{legacy_plugin_name}_config.json")
    if not legacy_path.is_file():
        return False

    try:
        with legacy_path.open(encoding="utf-8-sig") as legacy_file:
            legacy_config = json.load(legacy_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    if not isinstance(legacy_config, dict):
        return False

    merged = _merge_config(dict(config), legacy_config)
    config.clear()
    config.update(merged)
    config.save_config()
    return True
