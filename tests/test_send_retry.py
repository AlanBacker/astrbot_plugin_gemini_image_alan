from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
import time
import types
import unittest
from pathlib import Path

SEND_FAILED_TEXT = "❌ 图片已生成，但发送失败（可能是网络问题），请稍后重试"
UNEXPECTED_ERROR_TEXT = (
    "❌ 生成过程中出现意外错误，请稍后再试；一直出现的话请联系管理员查看后台日志"
)


class FakeChain:
    def __init__(self):
        self.images = []
        self.texts = []

    def message(self, text):
        self.texts.append(text)
        return self

    def file_image(self, path):
        self.images.append(path)
        return self


def build_plugin_class(sleeps: list[float]):
    """从 main.py 抽出发送相关的方法，组成可直接运行的测试类"""
    module = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    plugin_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GeminiImagePlugin"
    )
    wanted = {
        "SEND_RETRY_ATTEMPTS",
        "SEND_RETRY_DELAY",
        "_send_image_with_retry",
        "_generate_and_send_image_async",
    }
    body = [
        node
        for node in plugin_class.body
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name in wanted
            or isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in wanted for t in node.targets)
        )
    ]
    harness = ast.ClassDef(
        name="PluginHarness", bases=[], keywords=[], body=body, decorator_list=[]
    )

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    namespace = {
        "asyncio": types.SimpleNamespace(sleep=fake_sleep),
        "hashlib": hashlib,
        "time": time,
        "MessageChain": FakeChain,
        "save_temp_img": lambda data: f"/tmp/{data.decode()}.png",
        "logger": logging.getLogger("gemini_image_send_test"),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[])),
            "<send-harness>",
            "exec",
        ),
        namespace,
    )
    return namespace["PluginHarness"]


class SendRetryTests(unittest.IsolatedAsyncioTestCase):
    def make_plugin(self, image_failures: int):
        """前 image_failures 次发送图片失败，文字消息总是成功"""
        self.sleeps = []
        self.sent = []
        self.semaphore_held = []
        self.finished = []
        self.recorded = []
        plugin = build_plugin_class(self.sleeps)()
        plugin._generation_semaphore = asyncio.Semaphore(1)
        plugin.notify_rotate_retry = False
        plugin.show_grounding_sources = True
        test = self

        class FakeGenerator:
            result = ([b"cat"], None, [])

            async def generate_image(self, **kwargs):
                if isinstance(self.result, Exception):
                    raise self.result
                return self.result

        class FakeContext:
            async def send_message(self, origin, chain):
                if chain.images:
                    test.semaphore_held.append(plugin._generation_semaphore.locked())
                    attempts = sum(1 for kind, _ in test.sent if kind == "image")
                    test.sent.append(("image", chain.images))
                    if attempts < image_failures:
                        raise RuntimeError("highway upload transport failed")
                else:
                    test.sent.append(("text", chain.texts[0]))

        async def finish(subject, request_id, successful):
            test.finished.append(successful)

        async def quota_note(subject, cost):
            return self.quota_note

        self.quota_note = None
        plugin.generator = FakeGenerator()
        plugin._format_remaining_quota = quota_note
        plugin.context = FakeContext()
        plugin._finish_rate_limit_request = finish
        plugin._record_session_success = self.recorded.append
        return plugin

    async def run_task(self, plugin):
        with self.assertLogs("gemini_image_send_test", level="WARNING") as logs:
            await plugin._generate_and_send_image_async(
                "画猫",
                "qq:GroupMessage:1",
                session_subject_id="group:1",
                rate_limit_subject_id="group:1",
                rate_limit_request_id="req",
            )
        return logs.output

    async def test_image_is_resent_after_transient_failures(self):
        plugin = self.make_plugin(image_failures=2)

        await self.run_task(plugin)

        self.assertEqual([kind for kind, _ in self.sent], ["image"] * 3)
        self.assertEqual(self.sleeps, [3, 6])
        self.assertEqual(self.finished, [True])
        self.assertEqual(self.recorded, ["group:1"])

    async def test_user_is_told_image_could_not_be_sent(self):
        plugin = self.make_plugin(image_failures=99)

        logs = await self.run_task(plugin)

        self.assertEqual(
            self.sent,
            [("image", ["/tmp/cat.png"])] * 3 + [("text", SEND_FAILED_TEXT)],
        )
        # 没发出去不扣额度，也不算成功次数
        self.assertEqual(self.finished, [False])
        self.assertEqual(self.recorded, [])
        self.assertFalse(any("任务失败" in line for line in logs))

    async def test_sending_does_not_hold_generation_slot(self):
        plugin = self.make_plugin(image_failures=1)

        await self.run_task(plugin)

        self.assertEqual(self.semaphore_held, [False, False])

    async def test_sources_and_remaining_quota_follow_image(self):
        plugin = self.make_plugin(image_failures=0)
        plugin.generator.result = ([b"cat"], None, [("天气网", "https://w.example")])
        plugin._format_grounding_sources = lambda sources: "🔎 参考来源:\n1. 天气网"
        self.quota_note = "📊 本次消耗 2 次额度，本群剩余 7/10 次（滚动24小时）"
        texts = []

        async def send(origin, chain):
            texts.append(chain.texts)

        plugin.context.send_message = send

        await plugin._generate_and_send_image_async(
            "画猫",
            "qq:GroupMessage:1",
            rate_limit_subject_id="group:1",
            rate_limit_request_id="req",
            quota_cost=2,
        )

        self.assertEqual(
            texts,
            [
                [
                    "🔎 参考来源:\n1. 天气网\n📊 本次消耗 2 次额度，本群剩余 7/10 次（滚动24小时）"
                ]
            ],
        )

    async def test_generation_error_is_shown_with_readable_reason(self):
        plugin = self.make_plugin(image_failures=0)
        plugin.generator.result = (None, "请求太频繁或 API 额度已用完，请稍后再试", [])

        await plugin._generate_and_send_image_async(
            "画猫",
            "qq:GroupMessage:1",
            rate_limit_subject_id="group:1",
            rate_limit_request_id="req",
        )

        self.assertEqual(
            self.sent,
            [("text", "❌ 图片生成失败：请求太频繁或 API 额度已用完，请稍后再试")],
        )
        self.assertEqual(self.finished, [False])

    async def test_unexpected_error_tells_user_to_retry_or_ask_admin(self):
        plugin = self.make_plugin(image_failures=0)
        plugin.generator.result = KeyError("boom")

        with self.assertLogs("gemini_image_send_test", level="ERROR"):
            await plugin._generate_and_send_image_async(
                "画猫",
                "qq:GroupMessage:1",
                rate_limit_subject_id="group:1",
                rate_limit_request_id="req",
            )

        self.assertEqual(self.sent, [("text", UNEXPECTED_ERROR_TEXT)])
        self.assertEqual(self.finished, [False])


if __name__ == "__main__":
    unittest.main()
