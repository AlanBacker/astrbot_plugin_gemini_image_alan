from __future__ import annotations

import ast
import base64
import json
import logging
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

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
    model: str,
    api_type: str = "gemini",
    search_types: tuple[str, ...] = (),
    rotate_retry: bool = True,
) -> GeminiImageGenerator:
    return GeminiImageGenerator(
        api_keys=["key"],
        base_url="https://generativelanguage.googleapis.com",
        model=model,
        api_type=api_type,
        search_types=search_types,
        rotate_retry=rotate_retry,
    )


def make_marked_image() -> Image.Image:
    """左上角有红块、其余是蓝色的 40x20 图片，用来判断方向"""
    img = Image.new("RGB", (40, 20), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 10, 10))
    return img


def make_marked_png() -> bytes:
    output = BytesIO()
    make_marked_image().save(output, format="PNG")
    return output.getvalue()


def red_corner(image_data: bytes) -> str:
    with Image.open(BytesIO(image_data)) as img:
        rgb = img.convert("RGB")
        corners = {
            "左上": (5, 5),
            "右上": (rgb.width - 6, 5),
            "左下": (5, rgb.height - 6),
            "右下": (rgb.width - 6, rgb.height - 6),
        }
        return next(name for name, xy in corners.items() if rgb.getpixel(xy)[0] > 200)


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

    def test_sources_are_only_sent_when_display_is_enabled(self):
        module = ast.parse(Path("main.py").read_text(encoding="utf-8"))
        guards = [
            ast.unparse(node.test)
            for node in ast.walk(module)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Attribute)
                and child.attr == "_format_grounding_sources"
                for child in ast.walk(node)
            )
        ]

        self.assertEqual(guards, ["sources and self.show_grounding_sources"])

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


class RotateRetryTests(unittest.IsolatedAsyncioTestCase):
    def make_counting_generator(self, succeed_on: int | None, **kwargs):
        """前 succeed_on 次请求失败，之后原样返回收到的第一张参考图"""
        generator = make_generator("gemini-nano-banana-2.1", **kwargs)
        received = []

        async def fake_try(prompt, images_data, *args):
            received.append(images_data)
            if succeed_on is not None and len(received) > succeed_on:
                return [images_data[0][0]], None, [("来源", "https://a.example")]
            return None, f"第{len(received)}次失败", []

        async def no_wait(attempt):
            return None

        generator._try_generate_with_current_key = fake_try
        generator._wait_before_retry = no_wait
        return generator, received

    async def test_rotated_retry_succeeds_and_turns_result_back(self):
        generator, received = self.make_counting_generator(succeed_on=3)
        image = make_marked_png()

        images, error, sources = await generator.generate_image(
            "改成水彩", [(image, "image/png")]
        )

        self.assertIsNone(error)
        self.assertEqual(len(received), 4)
        self.assertTrue(all(r[0][0] == image for r in received[:3]))
        # 第4次请求收到的是旋转后的参考图，红点转到了右下角
        self.assertEqual(received[3][0][1], "image/png")
        self.assertEqual(red_corner(received[3][0][0]), "右下")
        # 返回给用户的图片已经转回来
        self.assertEqual(red_corner(images[0]), "左上")
        self.assertEqual(sources, [("来源", "https://a.example")])

    async def test_rotated_retry_uses_same_attempt_count(self):
        generator, received = self.make_counting_generator(succeed_on=None)

        images, error, _ = await generator.generate_image(
            "改成水彩", [(make_marked_png(), "image/png")]
        )

        self.assertIsNone(images)
        self.assertEqual(len(received), 6)
        self.assertEqual(error, "重试失败: 第6次失败")

    async def test_no_rotated_retry_when_disabled_or_text_only(self):
        generator, received = self.make_counting_generator(
            succeed_on=None, rotate_retry=False
        )
        _, error, _ = await generator.generate_image(
            "改成水彩", [(make_marked_png(), "image/png")]
        )
        self.assertEqual(len(received), 3)
        self.assertEqual(error, "重试失败: 第3次失败")

        generator, received = self.make_counting_generator(succeed_on=None)
        await generator.generate_image("画猫")
        self.assertEqual(len(received), 3)

    async def test_unreadable_reference_image_skips_rotated_retry(self):
        generator, received = self.make_counting_generator(succeed_on=None)

        with self.assertLogs("gemini_image_test", level="WARNING"):
            _, error, _ = await generator.generate_image(
                "改成水彩", [(b"RIFF\x00\x00\x00\x00WEBPbroken", "image/webp")]
            )

        self.assertEqual(len(received), 3)
        self.assertEqual(error, "重试失败: 第3次失败")

    def test_rotation_keeps_jpeg_and_follows_exif_orientation(self):
        exif = Image.Exif()
        exif[0x0112] = 3  # 显示时需要旋转 180 度
        output = BytesIO()
        make_marked_image().save(output, format="JPEG", quality=95, exif=exif)

        rotated, mime_type = GeminiImageGenerator._rotate_image_180(output.getvalue())

        self.assertEqual(mime_type, "image/jpeg")
        with Image.open(BytesIO(rotated)) as result:
            self.assertEqual(result.size, (40, 20))
            self.assertNotIn(0x0112, result.getexif())
        # 用户看到的是红块在右下角的图，旋转 180 度后回到左上角
        self.assertEqual(red_corner(rotated), "左上")

    def test_rotate_retry_option_defaults_to_on(self):
        schema = json.loads(Path("_conf_schema.json").read_text(encoding="utf-8-sig"))
        option = schema["generate_config"]["items"]["rotate_retry"]
        self.assertEqual(option["type"], "bool")
        self.assertIs(option["default"], True)

        module = ast.parse(Path("main.py").read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GeminiImageGenerator"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn(
            "rotate_retry=self.rotate_retry",
            [ast.unparse(k) for k in calls[0].keywords],
        )


if __name__ == "__main__":
    unittest.main()
