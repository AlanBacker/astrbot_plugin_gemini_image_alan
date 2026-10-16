from __future__ import annotations

import ast
import hashlib
import json
import logging
import tempfile
import time
import unittest
from pathlib import Path

from rate_limit import RateLimitStore


def build_plugin_class(*names: str):
    """从 main.py 抽出指定的方法和类属性，组成可直接运行的测试类"""
    module = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    plugin_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GeminiImagePlugin"
    )
    body = []
    for node in plugin_class.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in names:
                continue
            node.decorator_list = [
                d
                for d in node.decorator_list
                if isinstance(d, ast.Name) and d.id == "staticmethod"
            ]
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in names for t in node.targets
        ):
            body.append(node)
    harness = ast.ClassDef(
        name="PluginHarness", bases=[], keywords=[], body=body, decorator_list=[]
    )
    namespace = {
        "Any": object,
        "AstrMessageEvent": object,
        "hashlib": hashlib,
        "json": json,
        "time": time,
        "logger": logging.getLogger("gemini_image_hd_test"),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[])),
            "<hd-harness>",
            "exec",
        ),
        namespace,
    )
    return namespace["PluginHarness"]


class HdSettingsTests(unittest.TestCase):
    def setUp(self):
        self.plugin = build_plugin_class(
            "HD_RESOLUTIONS", "_get_hd_settings", "_positive_int"
        )()

    def test_defaults_to_2k_and_double_cost(self):
        self.plugin.config = {"generate_config": {}}
        self.assertEqual(self.plugin._get_hd_settings(), ("2K", 2))

    def test_custom_and_invalid_values(self):
        self.plugin.config = {
            "generate_config": {"hd_resolution": "4k", "hd_quota_cost": 3}
        }
        self.assertEqual(self.plugin._get_hd_settings(), ("4K", 3))

        self.plugin.config = {
            "generate_config": {"hd_resolution": "8K", "hd_quota_cost": "abc"}
        }
        self.assertEqual(self.plugin._get_hd_settings(), ("2K", 2))

        self.plugin.config = {"generate_config": {"hd_quota_cost": 0}}
        self.assertEqual(self.plugin._get_hd_settings(), ("2K", 1))

    def test_schema_defaults(self):
        schema = json.loads(Path("_conf_schema.json").read_text(encoding="utf-8-sig"))
        items = schema["generate_config"]["items"]
        self.assertEqual(items["hd_resolution"]["default"], "2K")
        self.assertEqual(items["hd_quota_cost"]["default"], 2)
        self.assertIs(items["show_remaining_quota"]["default"], True)
        self.assertEqual(items["default_resolution"]["default"], "1K")


class RemainingQuotaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.plugin = build_plugin_class(
            "_format_remaining_quota",
            "_get_rate_limit_settings",
            "_positive_int",
            "_as_bool",
        )()
        self.plugin.rate_limit_store = RateLimitStore(Path(self._temp_dir.name))
        self.plugin.config = {
            "generate_config": {
                "enable_rate_limit": True,
                "max_requests_per_minute": 100,
                "max_requests_per_hour": 100,
                "max_requests_per_day": 10,
            }
        }

    async def asyncTearDown(self):
        self._temp_dir.cleanup()

    async def reserve(self, subject, request_id, cost=1):
        allowed, message = await self.plugin.rate_limit_store.reserve(
            subject,
            request_id,
            enabled=True,
            minute_limit=100,
            hour_limit=100,
            day_limit=10,
            cost=cost,
        )
        self.assertTrue(allowed, message)

    async def test_current_task_is_counted_as_used(self):
        for index in range(3):
            await self.reserve("group:1", f"n{index}")
            await self.plugin.rate_limit_store.finish(
                "group:1", f"n{index}", successful=True
            )
        await self.reserve("group:1", "hd", cost=2)

        self.assertEqual(
            await self.plugin._format_remaining_quota("group:1", 2),
            "📊 本次消耗 2 次额度，本群剩余 5/10 次（滚动24小时）",
        )

        await self.reserve("user", "n")
        self.assertEqual(
            await self.plugin._format_remaining_quota("user", 1),
            "📊 本次消耗 1 次额度，剩余 9/10 次（滚动24小时）",
        )

    async def test_hidden_when_disabled(self):
        await self.reserve("user", "n")
        self.plugin.config["generate_config"]["show_remaining_quota"] = False
        self.assertIsNone(await self.plugin._format_remaining_quota("user", 1))

        self.plugin.config["generate_config"]["show_remaining_quota"] = True
        self.plugin.config["generate_config"]["enable_rate_limit"] = False
        self.assertIsNone(await self.plugin._format_remaining_quota("user", 1))


class FakeEvent:
    unified_msg_origin = "qq:GroupMessage:1"

    def __init__(self, text: str):
        self.message_str = text

    @staticmethod
    def plain_result(message):
        return message


class CommandFlowTests(unittest.IsolatedAsyncioTestCase):
    def make_plugin(self, allowed: bool = True):
        plugin = build_plugin_class(
            "HD_RESOLUTIONS",
            "generate_image_command",
            "generate_hd_image_command",
            "_run_generate_command",
            "_get_hd_settings",
            "_positive_int",
            "_get_rate_limit_subject",
        )()
        plugin.config = {"generate_config": {}}
        plugin.presets = {
            "表情包": json.dumps({"prompt": "画表情包", "resolution": "4K"})
        }
        plugin.default_aspect_ratio = "1:1"
        plugin.default_resolution = "1K"
        plugin.thinking_level = "默认"
        plugin.perm_silent = False
        self.reserved = []
        self.tasks = []

        async def no_images(event):
            return []

        async def reserve(subject, request_id, cost=1):
            self.reserved.append((subject, cost))
            return (
                allowed,
                "❌ 额度不足：本次需要 2 次额度，滚动24小时内只剩 1 次",
                True,
            )

        def generate(**kwargs):
            return kwargs

        plugin._get_event_user_id = lambda event: "u1"
        plugin._get_event_group_id = lambda event: "g1"
        plugin._check_permission = lambda user_id, group_id: True
        plugin._get_reference_images_for_command = no_images
        plugin._register_session = lambda *args: None
        plugin._reserve_rate_limit = reserve
        plugin._generate_and_send_image_async = generate
        plugin.create_background_task = self.tasks.append
        return plugin

    async def run_command(self, plugin, command, text):
        return [result async for result in getattr(plugin, command)(FakeEvent(text))]

    async def test_hd_command_uses_hd_resolution_and_cost(self):
        plugin = self.make_plugin()

        replies = await self.run_command(
            plugin, "generate_hd_image_command", "高清生图 一只猫"
        )

        self.assertEqual(replies, ["已开始高清生图任务[2K]"])
        self.assertEqual(self.reserved, [("group:g1", 2)])
        self.assertEqual(self.tasks[0]["prompt"], "一只猫")
        self.assertEqual(self.tasks[0]["resolution"], "2K")
        self.assertEqual(self.tasks[0]["quota_cost"], 2)

    async def test_hd_command_overrides_preset_resolution(self):
        plugin = self.make_plugin()
        plugin.config = {"generate_config": {"hd_resolution": "4K", "hd_quota_cost": 4}}

        await self.run_command(plugin, "generate_hd_image_command", "高清生图 表情包")
        self.assertEqual(self.tasks[0]["resolution"], "4K")
        self.assertEqual(self.tasks[0]["quota_cost"], 4)

        plugin.config = {"generate_config": {}}
        await self.run_command(plugin, "generate_hd_image_command", "高清生图 表情包")
        self.assertEqual(self.tasks[1]["resolution"], "2K")

    async def test_normal_command_keeps_default_cost(self):
        plugin = self.make_plugin()

        replies = await self.run_command(
            plugin, "generate_image_command", "生图 一只猫"
        )
        await self.run_command(plugin, "generate_image_command", "生图 表情包")

        self.assertEqual(replies, ["已开始生图任务"])
        self.assertEqual(self.reserved, [("group:g1", 1), ("group:g1", 1)])
        self.assertEqual(self.tasks[0]["resolution"], "1K")
        self.assertEqual(self.tasks[0]["quota_cost"], 1)
        # 普通生图仍按预设里的分辨率
        self.assertEqual(self.tasks[1]["resolution"], "4K")

    async def test_insufficient_quota_stops_before_generating(self):
        plugin = self.make_plugin(allowed=False)

        replies = await self.run_command(
            plugin, "generate_hd_image_command", "高清生图 一只猫"
        )

        self.assertEqual(
            replies, ["❌ 额度不足：本次需要 2 次额度，滚动24小时内只剩 1 次"]
        )
        self.assertEqual(self.tasks, [])


if __name__ == "__main__":
    unittest.main()
