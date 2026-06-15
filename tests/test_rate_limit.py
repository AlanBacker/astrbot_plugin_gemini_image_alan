from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from rate_limit import RateLimitStore


class RateLimitStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_generation_does_not_consume_quota(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RateLimitStore(Path(temp_dir))

            allowed, _ = await store.reserve(
                "user",
                "failed",
                enabled=True,
                minute_limit=2,
                hour_limit=20,
                day_limit=20,
                now=1000,
            )
            self.assertTrue(allowed)
            self.assertTrue(
                await store.finish("user", "failed", successful=False, now=1001)
            )

            snapshot, error = await store.snapshot("user", now=1001)
            self.assertEqual(error, "")
            self.assertEqual(snapshot["day"], 0)
            self.assertEqual(snapshot["pending"], 0)

    async def test_successful_quota_survives_restart_and_blocks_request_21(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            store = RateLimitStore(data_dir)

            for index in range(20):
                request_id = f"success-{index}"
                allowed, message = await store.reserve(
                    "user",
                    request_id,
                    enabled=True,
                    minute_limit=100,
                    hour_limit=100,
                    day_limit=20,
                    now=1000 + index,
                )
                self.assertTrue(allowed, message)
                self.assertTrue(
                    await store.finish(
                        "user",
                        request_id,
                        successful=True,
                        now=1000 + index,
                    )
                )

            restarted_store = RateLimitStore(data_dir)
            allowed, message = await restarted_store.reserve(
                "user",
                "success-21",
                enabled=True,
                minute_limit=100,
                hour_limit=100,
                day_limit=20,
                now=2000,
            )
            self.assertFalse(allowed)
            self.assertIn("滚动24小时限 20 次", message)

    async def test_pending_reservation_is_shared_between_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            first_store = RateLimitStore(data_dir)
            second_store = RateLimitStore(data_dir)

            allowed, _ = await first_store.reserve(
                "user",
                "first",
                enabled=True,
                minute_limit=1,
                hour_limit=1,
                day_limit=1,
                now=3000,
            )
            self.assertTrue(allowed)

            allowed, message = await second_store.reserve(
                "user",
                "second",
                enabled=True,
                minute_limit=1,
                hour_limit=1,
                day_limit=1,
                now=3000,
            )
            self.assertFalse(allowed)
            self.assertIn("每分钟限 1 次", message)

    async def test_corrupt_storage_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "rate_limit_usage.json").write_text(
                "not-json", encoding="utf-8"
            )
            store = RateLimitStore(data_dir)

            allowed, message = await store.reserve(
                "user",
                "request",
                enabled=True,
                minute_limit=2,
                hour_limit=20,
                day_limit=20,
                now=3500,
            )
            self.assertFalse(allowed)
            self.assertIn("已停止生图以防止超额消费", message)

    async def test_rolling_day_expires_after_24_hours(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RateLimitStore(Path(temp_dir))
            self.assertTrue(
                (
                    await store.reserve(
                        "user",
                        "success",
                        enabled=True,
                        minute_limit=10,
                        hour_limit=10,
                        day_limit=1,
                        now=4000,
                    )
                )[0]
            )
            await store.finish("user", "success", successful=True, now=4000)

            self.assertFalse(
                (
                    await store.reserve(
                        "user",
                        "before-expiry",
                        enabled=True,
                        minute_limit=10,
                        hour_limit=10,
                        day_limit=1,
                        now=4000 + 86399,
                    )
                )[0]
            )
            self.assertTrue(
                (
                    await store.reserve(
                        "user",
                        "at-expiry",
                        enabled=True,
                        minute_limit=10,
                        hour_limit=10,
                        day_limit=1,
                        now=4000 + 86400,
                    )
                )[0]
            )


class EntrypointCoverageTests(unittest.TestCase):
    @staticmethod
    def _plugin_method_harness(*method_names: str):
        source = Path("main.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        plugin_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "GeminiImagePlugin"
        )
        methods = [
            node
            for node in plugin_class.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        ]
        harness = ast.ClassDef(
            name="PluginHarness",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        ast.fix_missing_locations(harness)
        namespace = {"Any": object}
        exec(
            compile(
                ast.Module(body=[harness], type_ignores=[]),
                "<plugin-harness>",
                "exec",
            ),
            namespace,
        )
        return namespace["PluginHarness"]

    def test_webui_rate_limit_changes_are_read_on_every_request(self):
        harness_class = self._plugin_method_harness(
            "_positive_int", "_as_bool", "_get_rate_limit_settings"
        )
        plugin = harness_class()
        plugin.config = {
            "generate_config": {
                "enable_rate_limit": True,
                "max_requests_per_minute": 3,
                "max_requests_per_hour": 30,
                "max_requests_per_day": 100,
            }
        }
        self.assertEqual(plugin._get_rate_limit_settings(), (True, 3, 30, 100))

        plugin.config["generate_config"]["max_requests_per_day"] = 20
        self.assertEqual(plugin._get_rate_limit_settings(), (True, 3, 30, 20))

    def test_command_and_llm_tool_both_reserve_quota(self):
        source = Path("main.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        tool_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "GeminiImageGenerationTool"
        )
        plugin_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "GeminiImagePlugin"
        )
        tool_call = next(
            node
            for node in tool_class.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "call"
        )
        command_call = next(
            node
            for node in plugin_class.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "generate_image_command"
        )

        for entrypoint in (tool_call, command_call):
            awaited_methods = {
                node.value.func.attr
                for node in ast.walk(entrypoint)
                if isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
            }
            self.assertIn("_reserve_rate_limit", awaited_methods)


if __name__ == "__main__":
    unittest.main()
