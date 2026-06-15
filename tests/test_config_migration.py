from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config_migration import migrate_legacy_config


CURRENT_PLUGIN_NAME = "astrbot_plugin_gemini_image_alan"
LEGACY_PLUGIN_NAME = "astrbot_plugin_gemini_image"


class FakeConfig(dict):
    def __init__(self, config_path: Path, values: dict, *, first_deploy: bool):
        super().__init__(values)
        self.config_path = str(config_path)
        self.first_deploy = first_deploy
        self.save_count = 0

    def save_config(self):
        self.save_count += 1


class ConfigMigrationTests(unittest.TestCase):
    def test_migrates_legacy_config_and_preserves_new_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            legacy_path = tmp_path / f"{LEGACY_PLUGIN_NAME}_config.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "api_config": {
                            "api_key": ["secret"],
                            "provider_id": "provider-1",
                        },
                        "generate_config": {"max_requests_per_day": 20},
                        "permission_config": {
                            "mode": "whitelist",
                            "users": ["123"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = FakeConfig(
                tmp_path / f"{CURRENT_PLUGIN_NAME}_config.json",
                {
                    "api_config": {
                        "api_key": [],
                        "provider_id": "",
                        "new_option": True,
                    },
                    "generate_config": {
                        "max_requests_per_day": 100,
                        "max_concurrent_generations": 3,
                    },
                    "permission_config": {
                        "mode": "disable",
                        "users": [],
                        "groups": [],
                    },
                },
                first_deploy=True,
            )

            migrated = migrate_legacy_config(
                config,
                current_plugin_name=CURRENT_PLUGIN_NAME,
                legacy_plugin_name=LEGACY_PLUGIN_NAME,
            )

            self.assertTrue(migrated)
            self.assertEqual(
                config["api_config"],
                {
                    "api_key": ["secret"],
                    "provider_id": "provider-1",
                    "new_option": True,
                },
            )
            self.assertEqual(
                config["generate_config"],
                {
                    "max_requests_per_day": 20,
                    "max_concurrent_generations": 3,
                },
            )
            self.assertEqual(
                config["permission_config"],
                {
                    "mode": "whitelist",
                    "users": ["123"],
                    "groups": [],
                },
            )
            self.assertEqual(config.save_count, 1)

    def test_does_not_overwrite_an_existing_fork_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            legacy_path = tmp_path / f"{LEGACY_PLUGIN_NAME}_config.json"
            legacy_path.write_text(
                '{"api_config": {"api_key": ["legacy"]}}',
                encoding="utf-8",
            )
            config = FakeConfig(
                tmp_path / f"{CURRENT_PLUGIN_NAME}_config.json",
                {"api_config": {"api_key": ["current"]}},
                first_deploy=False,
            )

            migrated = migrate_legacy_config(
                config,
                current_plugin_name=CURRENT_PLUGIN_NAME,
                legacy_plugin_name=LEGACY_PLUGIN_NAME,
            )

            self.assertFalse(migrated)
            self.assertEqual(config["api_config"]["api_key"], ["current"])
            self.assertEqual(config.save_count, 0)


if __name__ == "__main__":
    unittest.main()
