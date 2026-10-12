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
            async def generate_image(self, **kwargs):
                return [b"cat"], None, []

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

        plugin.generator = FakeGenerator()
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


if __name__ == "__main__":
    unittest.main()
