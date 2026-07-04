from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_migration import LEGACY_DATA_MIGRATION_MARKER, migrate_legacy_data_dir


class DataMigrationTests(unittest.TestCase):
    def test_clean_install_uses_current_dir_without_creating_legacy_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_dir = root / "plugin_data" / "astrbot_plugin_gemini_image_alan"
            legacy_dir = root / "plugin_data" / "astrbot_plugin_gemini_image"

            migrated = migrate_legacy_data_dir(current_dir, legacy_dir)

            self.assertFalse(migrated)
            self.assertTrue(current_dir.is_dir())
            self.assertTrue((current_dir / LEGACY_DATA_MIGRATION_MARKER).is_file())
            self.assertFalse(legacy_dir.exists())

    def test_copies_legacy_data_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_dir = root / "plugin_data" / "astrbot_plugin_gemini_image_alan"
            legacy_dir = root / "plugin_data" / "astrbot_plugin_gemini_image"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "rate_limit_usage.json").write_text(
                '{"version":1}',
                encoding="utf-8",
            )
            (legacy_dir / "rate_limit_usage.lock").write_text("", encoding="utf-8")
            (legacy_dir / "rate_limit_usage.json.bak_before_reset").write_text(
                '{"version":"backup"}',
                encoding="utf-8",
            )
            nested_dir = legacy_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "data.json").write_text('{"ok":true}', encoding="utf-8")

            migrated = migrate_legacy_data_dir(current_dir, legacy_dir)

            self.assertTrue(migrated)
            self.assertEqual(
                (current_dir / "rate_limit_usage.json").read_text(encoding="utf-8"),
                '{"version":1}',
            )
            self.assertFalse((current_dir / "rate_limit_usage.lock").exists())
            self.assertFalse(
                (current_dir / "rate_limit_usage.json.bak_before_reset").exists()
            )
            self.assertEqual(
                (current_dir / "nested" / "data.json").read_text(encoding="utf-8"),
                '{"ok":true}',
            )

            (legacy_dir / "rate_limit_usage.json").write_text(
                '{"version":2}',
                encoding="utf-8",
            )
            migrated_again = migrate_legacy_data_dir(current_dir, legacy_dir)

            self.assertFalse(migrated_again)
            self.assertEqual(
                (current_dir / "rate_limit_usage.json").read_text(encoding="utf-8"),
                '{"version":1}',
            )

    def test_does_not_overwrite_existing_current_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_dir = root / "plugin_data" / "astrbot_plugin_gemini_image_alan"
            legacy_dir = root / "plugin_data" / "astrbot_plugin_gemini_image"
            current_dir.mkdir(parents=True)
            legacy_dir.mkdir(parents=True)
            (current_dir / "rate_limit_usage.json").write_text(
                '{"version":"current"}',
                encoding="utf-8",
            )
            (legacy_dir / "rate_limit_usage.json").write_text(
                '{"version":"legacy"}',
                encoding="utf-8",
            )

            migrated = migrate_legacy_data_dir(current_dir, legacy_dir)

            self.assertFalse(migrated)
            self.assertEqual(
                (current_dir / "rate_limit_usage.json").read_text(encoding="utf-8"),
                '{"version":"current"}',
            )
            self.assertTrue((current_dir / LEGACY_DATA_MIGRATION_MARKER).is_file())


if __name__ == "__main__":
    unittest.main()
