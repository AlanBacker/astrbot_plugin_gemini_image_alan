from __future__ import annotations

import shutil
from pathlib import Path

LEGACY_DATA_MIGRATION_MARKER = ".legacy_data_migrated"


def _should_skip_legacy_data(path: Path) -> bool:
    return (
        path.name == LEGACY_DATA_MIGRATION_MARKER
        or path.suffix == ".lock"
        or path.suffix == ".bak"
        or path.suffix.startswith(".bak_")
    )


def _copy_missing_legacy_data(source: Path, target: Path) -> bool:
    if _should_skip_legacy_data(source):
        return False

    if source.is_dir():
        copied = False
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            copied = _copy_missing_legacy_data(child, target / child.name) or copied
        return copied

    if source.is_file():
        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return True

    return False


def migrate_legacy_data_dir(current_dir: Path, legacy_dir: Path) -> bool:
    """Copy legacy plugin data into the current plugin data dir exactly once."""
    current_dir = Path(current_dir)
    legacy_dir = Path(legacy_dir)
    marker_path = current_dir / LEGACY_DATA_MIGRATION_MARKER

    current_dir.mkdir(parents=True, exist_ok=True)
    if marker_path.exists():
        return False

    migrated = False
    if legacy_dir.is_dir() and legacy_dir.resolve() != current_dir.resolve():
        for item in legacy_dir.iterdir():
            migrated = (
                _copy_missing_legacy_data(item, current_dir / item.name) or migrated
            )

    marker_path.write_text("done\n", encoding="utf-8")
    return migrated
