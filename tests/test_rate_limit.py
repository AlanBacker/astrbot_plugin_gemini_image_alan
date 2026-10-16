from __future__ import annotations

import ast
import asyncio
import copy
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

    async def test_hd_request_needs_enough_quota_for_its_cost(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RateLimitStore(Path(temp_dir))
            limits = {"enabled": True, "minute_limit": 100, "hour_limit": 100}

            # 先用掉 2 次：普通生图 1 次 + 失败的高清生图不计
            allowed, _ = await store.reserve(
                "user", "n1", day_limit=4, now=1000, **limits
            )
            self.assertTrue(allowed)
            await store.finish("user", "n1", successful=True, now=1000)
            allowed, _ = await store.reserve(
                "user", "hd-fail", day_limit=4, cost=2, now=1001, **limits
            )
            self.assertTrue(allowed)
            await store.finish("user", "hd-fail", successful=False, now=1002)
            snapshot, _ = await store.snapshot("user", now=1002)
            self.assertEqual(
                (snapshot["day"], snapshot["pending"], snapshot["reserved"]), (1, 0, 0)
            )

            # 剩 3 次时高清生图占 2 份，执行中只算 1 个任务
            allowed, _ = await store.reserve(
                "user", "hd", day_limit=4, cost=2, now=1003, **limits
            )
            self.assertTrue(allowed)
            snapshot, _ = await store.snapshot("user", now=1003)
            self.assertEqual(
                (snapshot["day"], snapshot["pending"], snapshot["reserved"]), (1, 1, 2)
            )

            # 只剩 1 份时，第二个高清生图被拒，普通生图还能用
            allowed, message = await store.reserve(
                "user", "hd2", day_limit=4, cost=2, now=1004, **limits
            )
            self.assertFalse(allowed)
            self.assertEqual(
                message, "❌ 额度不足：本次需要 2 次额度，滚动24小时内只剩 1 次"
            )

            self.assertTrue(await store.finish("user", "hd", successful=True, now=1005))
            snapshot, _ = await store.snapshot("user", now=1005)
            self.assertEqual(
                (snapshot["day"], snapshot["pending"], snapshot["reserved"]), (3, 0, 0)
            )

            allowed, _ = await store.reserve(
                "user", "n2", day_limit=4, now=1006, **limits
            )
            self.assertTrue(allowed)
            await store.finish("user", "n2", successful=True, now=1006)

            # 额度全部用完时，提示与普通生图相同
            allowed, message = await store.reserve(
                "user", "hd3", day_limit=4, cost=2, now=1007, **limits
            )
            self.assertFalse(allowed)
            self.assertIn("今日成功生图次数已达上限", message)

    async def test_hd_cost_also_counts_against_minute_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RateLimitStore(Path(temp_dir))
            limits = {
                "enabled": True,
                "minute_limit": 3,
                "hour_limit": 30,
                "day_limit": 100,
            }

            allowed, _ = await store.reserve("user", "hd", cost=2, now=1000, **limits)
            self.assertTrue(allowed)
            allowed, message = await store.reserve(
                "user", "hd2", cost=2, now=1001, **limits
            )
            self.assertFalse(allowed)
            self.assertEqual(
                message, "❌ 额度不足：本次需要 2 次额度，最近1分钟内只剩 1 次"
            )
            # 一分钟后高清生图的记录不再计入每分钟限制
            await store.finish("user", "hd", successful=True, now=1001)
            allowed, _ = await store.reserve("user", "hd3", cost=2, now=1062, **limits)
            self.assertTrue(allowed)

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

    async def test_private_user_and_group_have_independent_quota(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RateLimitStore(Path(temp_dir))

            allowed, _ = await store.reserve(
                "A",
                "private-success",
                enabled=True,
                minute_limit=10,
                hour_limit=10,
                day_limit=1,
                now=5000,
            )
            self.assertTrue(allowed)
            await store.finish("A", "private-success", successful=True, now=5000)

            allowed, _ = await store.reserve(
                "A",
                "private-blocked",
                enabled=True,
                minute_limit=10,
                hour_limit=10,
                day_limit=1,
                now=5001,
            )
            self.assertFalse(allowed)

            allowed, message = await store.reserve(
                "group:B",
                "group-success",
                enabled=True,
                minute_limit=10,
                hour_limit=10,
                day_limit=1,
                now=5001,
            )
            self.assertTrue(allowed, message)
            await store.finish("group:B", "group-success", successful=True, now=5001)

            private_snapshot, _ = await store.snapshot("A", now=5002)
            group_snapshot, _ = await store.snapshot("group:B", now=5002)
            self.assertEqual(private_snapshot["day"], 1)
            self.assertEqual(group_snapshot["day"], 1)


class EntrypointCoverageTests(unittest.TestCase):
    class _NullLogger:
        @staticmethod
        def info(*args, **kwargs):
            pass

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
            copy.deepcopy(node)
            for node in plugin_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in method_names
        ]
        for method in methods:
            method.decorator_list = [
                decorator
                for decorator in method.decorator_list
                if isinstance(decorator, ast.Name)
                and decorator.id in {"staticmethod", "classmethod"}
            ]
        harness = ast.ClassDef(
            name="PluginHarness",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        ast.fix_missing_locations(harness)
        namespace = {
            "Any": object,
            "AstrMessageEvent": object,
            "logger": EntrypointCoverageTests._NullLogger(),
        }
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

    def test_permissions_and_quota_follow_conversation_scope(self):
        harness_class = self._plugin_method_harness(
            "_check_permission", "_get_rate_limit_subject"
        )
        plugin = harness_class()
        plugin.config = {
            "permission_config": {
                "mode": "whitelist",
                "users": ["A"],
                "groups": ["B"],
            }
        }

        self.assertTrue(plugin._check_permission("A", ""))
        self.assertFalse(plugin._check_permission("C", ""))
        self.assertTrue(plugin._check_permission("A", "B"))
        self.assertTrue(plugin._check_permission("C", "B"))
        self.assertFalse(plugin._check_permission("A", "OTHER_GROUP"))
        self.assertEqual(plugin._get_rate_limit_subject("A", ""), "A")
        self.assertEqual(plugin._get_rate_limit_subject("A", "B"), "group:B")

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
            and node.name == "_run_generate_command"
        )
        # 生图和高清生图指令都走同一个流程
        for name in ("generate_image_command", "generate_hd_image_command"):
            command = next(
                node
                for node in plugin_class.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == name
            )
            self.assertIn(
                "_run_generate_command",
                {
                    node.func.attr
                    for node in ast.walk(command)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                },
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
            called_methods = {
                node.func.attr
                for node in ast.walk(entrypoint)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            self.assertIn("_get_rate_limit_subject", called_methods)

    def test_quota_query_checks_permission_and_uses_conversation_subject(self):
        source = Path("main.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        plugin_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "GeminiImagePlugin"
        )
        status_command = next(
            node
            for node in plugin_class.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "rate_limit_status_command"
        )
        called_methods = {
            node.func.attr
            for node in ast.walk(status_command)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_check_permission", called_methods)
        self.assertIn("_get_rate_limit_subject", called_methods)
        attribute_names = {
            node.attr
            for node in ast.walk(status_command)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn("perm_no_permission_reply", attribute_names)
        self.assertIn("plain_result", called_methods)

    def test_quota_query_denies_whitelisted_user_in_unlisted_group(self):
        harness_class = self._plugin_method_harness(
            "_check_permission",
            "_get_event_user_id",
            "_get_event_group_id",
            "_positive_int",
            "_as_bool",
            "_get_rate_limit_settings",
            "_get_rate_limit_subject",
            "rate_limit_status_command",
        )

        class FakeRateLimitStore:
            def __init__(self):
                self.snapshot_called = False

            async def snapshot(self, subject_id):
                self.snapshot_called = True
                return {}, ""

        class FakeEvent:
            unified_msg_origin = "fake-origin"
            message_obj = type(
                "MessageObject",
                (),
                {"group_id": "OTHER_GROUP", "sender": None},
            )()

            @staticmethod
            def get_sender_id():
                return "A"

            @staticmethod
            def plain_result(message):
                return message

        plugin = harness_class()
        plugin.config = {
            "permission_config": {
                "mode": "whitelist",
                "users": ["A"],
                "groups": ["B"],
            }
        }
        plugin.perm_no_permission_reply = "NO_PERMISSION"
        plugin.rate_limit_store = FakeRateLimitStore()

        async def collect_results():
            return [
                result async for result in plugin.rate_limit_status_command(FakeEvent())
            ]

        results = asyncio.run(collect_results())
        self.assertEqual(results, ["NO_PERMISSION"])
        self.assertFalse(plugin.rate_limit_store.snapshot_called)


if __name__ == "__main__":
    unittest.main()
