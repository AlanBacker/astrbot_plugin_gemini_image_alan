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

from gemini_generator import (
    GeminiImageGenerator,
    get_supported_search_types,
    get_supported_thinking_levels,
)


def make_generator(
    model: str, api_type: str = "gemini", search_types: tuple[str, ...] = ()
) -> GeminiImageGenerator:
    return GeminiImageGenerator(
        api_keys=["key"],
        base_url="https://generativelanguage.googleapis.com",
        model=model,
        api_type=api_type,
        search_types=search_types,
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


GROUNDED_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"inlineData": {"data": base64.b64encode(b"img").decode()}}]
            },
            "groundingMetadata": {
                "webSearchQueries": ["上海 天气"],
                "imageSearchQueries": ["Timareta butterfly"],
                "groundingChunks": [
                    {"web": {"uri": "https://weather.example/sh", "title": "天气网"}},
                    {
                        "image": {
                            "sourceUri": "https://bugs.example/timareta",
                            "imageUri": "https://bugs.example/timareta.jpg",
                            "domain": "bugs.example",
                        }
                    },
                    {"web": {"uri": "https://weather.example/sh", "title": "重复"}},
                    {"web": {"uri": "https://no-title.example"}},
                ],
            },
        }
    ]
}


class SearchGroundingTests(unittest.IsolatedAsyncioTestCase):
    def test_supported_search_types_follow_model(self):
        self.assertEqual(
            get_supported_search_types("gemini-nano-banana-2.1"), ("web", "image")
        )
        self.assertEqual(
            get_supported_search_types("gemini-3.1-flash-image"), ("web", "image")
        )
        self.assertEqual(
            get_supported_search_types("gemini-3-pro-image-preview"), ("web",)
        )
        self.assertEqual(get_supported_search_types("gemini-3.1-flash-lite-image"), ())
        self.assertEqual(get_supported_search_types("gemini-2.5-flash-image"), ())

    def test_payload_tools_for_each_search_mode(self):
        generator = make_generator("gemini-nano-banana-2.1")
        cases = {
            ("web",): [{"google_search": {}}],
            ("image",): [{"google_search": {"searchTypes": {"imageSearch": {}}}}],
            ("web", "image"): [
                {"google_search": {"searchTypes": {"webSearch": {}, "imageSearch": {}}}}
            ],
        }
        for search_types, tools in cases.items():
            payload = generator._build_gemini_payload(
                "cat", [], "1:1", "1K", None, search_types
            )
            self.assertEqual(payload["tools"], tools)

        payload = generator._build_gemini_payload("cat", [], "1:1", "1K")
        self.assertNotIn("tools", payload)

    def test_unsupported_search_types_are_dropped(self):
        generator = make_generator(
            "gemini-3-pro-image-preview", search_types=("web", "image")
        )
        with self.assertLogs("gemini_image_test", level="WARNING"):
            self.assertEqual(generator._resolve_search_types(), ("web",))

        generator = make_generator("gemini-2.5-flash-image", search_types=("web",))
        with self.assertLogs("gemini_image_test", level="WARNING"):
            self.assertEqual(generator._resolve_search_types(), ())

        generator = make_generator(
            "gemini-nano-banana-2.1", api_type="openai", search_types=("web",)
        )
        with self.assertLogs("gemini_image_test", level="WARNING"):
            self.assertEqual(generator._resolve_search_types(), ())

        self.assertEqual(
            make_generator("gemini-nano-banana-2.1")._resolve_search_types(), ()
        )

    async def test_gemini_returns_deduplicated_sources(self):
        generator = make_generator(
            "gemini-nano-banana-2.1", search_types=("web", "image")
        )
        sent_payloads = []

        async def fake_request(session, payload, task_id=None):
            sent_payloads.append(payload)
            return GROUNDED_RESPONSE

        generator._get_session = lambda: None
        generator._make_gemini_request = fake_request

        images, error, sources = await generator.generate_image("画蝴蝶")

        self.assertIsNone(error)
        self.assertEqual(images, [b"img"])
        self.assertEqual(
            sources,
            [
                ("天气网", "https://weather.example/sh"),
                ("bugs.example", "https://bugs.example/timareta"),
                ("", "https://no-title.example"),
            ],
        )
        self.assertIn("tools", sent_payloads[0])

    async def test_openai_result_has_empty_sources(self):
        generator = make_generator("gemini-nano-banana-2.1", api_type="openai")

        async def fake_openai(*args, **kwargs):
            return [b"img"], None

        generator._generate_openai = fake_openai

        self.assertEqual(await generator.generate_image("cat"), ([b"img"], None, []))

    def test_sources_are_formatted_as_links(self):
        module = ast.parse(Path("main.py").read_text(encoding="utf-8"))
        plugin_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "GeminiImagePlugin"
        )
        method = next(
            node
            for node in plugin_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_format_grounding_sources"
        )
        method.decorator_list = []
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                "<format-sources>",
                "exec",
            ),
            namespace,
        )

        text = namespace["_format_grounding_sources"](
            [("天气网", "https://weather.example/sh"), ("", "https://no-title.example")]
        )

        self.assertEqual(
            text,
            "🔎 参考来源:\n"
            "1. 天气网 https://weather.example/sh\n"
            "2. https://no-title.example",
        )


if __name__ == "__main__":
    unittest.main()
