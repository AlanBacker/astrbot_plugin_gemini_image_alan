from __future__ import annotations

import ast
import base64
import logging
import sys
import types
import unittest
from pathlib import Path

# 测试环境没有 AstrBot 时，用最小桩模块替代
if "astrbot.api" not in sys.modules:
    astrbot_module = types.ModuleType("astrbot")
    astrbot_api_module = types.ModuleType("astrbot.api")
    astrbot_api_module.logger = logging.getLogger("gemini_image_test")
    astrbot_module.api = astrbot_api_module
    sys.modules.setdefault("astrbot", astrbot_module)
    sys.modules.setdefault("astrbot.api", astrbot_api_module)

try:
    import aiohttp  # noqa: F401
except ImportError:
    sys.modules["aiohttp"] = types.ModuleType("aiohttp")

from gemini_generator import GeminiImageGenerator, get_supported_thinking_levels


def make_generator(model: str, api_type: str = "gemini") -> GeminiImageGenerator:
    return GeminiImageGenerator(
        api_keys=["key"],
        base_url="https://generativelanguage.googleapis.com",
        model=model,
        api_type=api_type,
    )


class ThinkingLevelTests(unittest.TestCase):
    def test_supported_levels_follow_model(self):
        self.assertEqual(
            get_supported_thinking_levels("gemini-nano-banana-2.1"),
            ("minimal", "medium", "high"),
        )
        self.assertEqual(
            get_supported_thinking_levels("gemini-3.1-flash-image"),
            ("minimal", "high"),
        )
        self.assertEqual(
            get_supported_thinking_levels("gemini-3.1-flash-lite-image"),
            ("minimal", "high"),
        )
        self.assertEqual(get_supported_thinking_levels("gemini-2.5-flash-image"), ())
        self.assertEqual(
            get_supported_thinking_levels("gemini-3-pro-image-preview"), ()
        )

    def test_nano_banana_2_1_payload_contains_thinking_level_and_size(self):
        generator = make_generator("gemini-nano-banana-2.1")
        level = generator._resolve_thinking_level("High")

        payload = generator._build_gemini_payload("cat", [], "16:9", "2K", level)

        self.assertEqual(
            payload["generationConfig"]["thinkingConfig"], {"thinkingLevel": "high"}
        )
        self.assertEqual(
            payload["generationConfig"]["imageConfig"],
            {"aspectRatio": "16:9", "imageSize": "2K"},
        )

    def test_unsupported_level_is_dropped(self):
        generator = make_generator("gemini-3.1-flash-image")
        with self.assertLogs("gemini_image_test", level="WARNING"):
            level = generator._resolve_thinking_level("medium")
        self.assertIsNone(level)

        payload = generator._build_gemini_payload("cat", [], "1:1", "1K", level)
        self.assertNotIn("thinkingConfig", payload["generationConfig"])

    def test_model_without_thinking_control_ignores_level(self):
        generator = make_generator("gemini-2.5-flash-image")
        with self.assertLogs("gemini_image_test", level="WARNING"):
            self.assertIsNone(generator._resolve_thinking_level("high"))
        self.assertIsNone(generator._resolve_thinking_level(None))

    def test_openai_payload_uses_reasoning_effort(self):
        generator = make_generator("gemini-nano-banana-2.1", api_type="openai")

        with_level = generator._build_openai_chat_payload(
            "cat", None, "1:1", "1K", "minimal"
        )
        without_level = generator._build_openai_chat_payload("cat", None, "1:1", "1K")

        self.assertEqual(with_level["reasoning_effort"], "minimal")
        self.assertNotIn("reasoning_effort", without_level)

    def test_thought_images_are_not_returned(self):
        generator = make_generator("gemini-nano-banana-2.1")
        draft = base64.b64encode(b"draft").decode()
        final = base64.b64encode(b"final").decode()
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "planning"},
                            {"thought": True, "inlineData": {"data": draft}},
                            {"inlineData": {"data": final}},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(generator._extract_gemini_image(response), [b"final"])

    def test_command_and_llm_tool_pass_thinking_level(self):
        module = ast.parse(Path("main.py").read_text(encoding="utf-8"))
        callers = []
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_generate_and_send_image_async"
            ):
                callers.append({keyword.arg for keyword in node.keywords})

        self.assertEqual(len(callers), 2)
        for keywords in callers:
            self.assertIn("thinking_level", keywords)


if __name__ == "__main__":
    unittest.main()
