from __future__ import annotations

import ast
import copy
import json
import re
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Any

from rate_limit import RateLimitStore
from session_registry import SessionRegistry, describe_subject

PLUGIN_NAME = "astrbot_plugin_gemini_image_alan"


async def add_successes(store: RateLimitStore, subject: str, times: list[float]):
    for index, now in enumerate(times):
        request_id = f"{subject}-{index}-{now}"
        allowed, message = await store.reserve(
            subject,
            request_id,
            enabled=True,
            minute_limit=10000,
            hour_limit=10000,
            day_limit=10000,
            now=now,
        )
        assert allowed, message
        assert await store.finish(subject, request_id, successful=True, now=now)


class RateLimitAdminTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._temp_dir.name)
        self.store = RateLimitStore(self.data_dir)

    async def asyncTearDown(self):
        self._temp_dir.cleanup()

    async def test_overview_lists_every_subject(self):
        await add_successes(self.store, "group:1", [1000, 1001, 8000])
        await self.store.reserve(
            "user",
            "pending",
            enabled=True,
            minute_limit=3,
            hour_limit=30,
            day_limit=100,
            now=8000,
        )

        overview, error = await self.store.overview(now=8010)

        self.assertEqual(error, "")
        self.assertEqual(
            overview["group:1"],
            {
                "minute": 1,
                "hour": 1,
                "day": 3,
                "pending": 0,
                "next_release": 1000 + RateLimitStore.WINDOW_SECONDS,
            },
        )
        self.assertEqual(overview["user"]["pending"], 1)
        self.assertEqual(overview["user"]["day"], 0)
        self.assertIsNone(overview["user"]["next_release"])

    async def test_lowering_count_drops_oldest_records_first(self):
        await add_successes(self.store, "user", [1000, 2000, 5000, 9000, 9030])

        ok, error = await self.store.set_used("user", 3, now=9040)

        self.assertTrue(ok, error)
        overview, _ = await self.store.overview(now=9040)
        self.assertEqual(overview["user"]["day"], 3)
        # 最近1分钟、1小时的记录保持不变
        self.assertEqual(overview["user"]["minute"], 2)
        self.assertEqual(overview["user"]["hour"], 2)
        self.assertEqual(
            overview["user"]["next_release"], 5000 + RateLimitStore.WINDOW_SECONDS
        )

    async def test_raising_count_only_affects_day_window(self):
        await add_successes(self.store, "user", [10000])

        ok, error = await self.store.set_used("user", 40, now=20000)

        self.assertTrue(ok, error)
        overview, _ = await self.store.overview(now=20000)
        self.assertEqual(overview["user"]["day"], 40)
        self.assertEqual(overview["user"]["minute"], 0)
        self.assertEqual(overview["user"]["hour"], 0)
        # 小时限额 30 不会因为补记 39 次而拦截
        allowed, message = await self.store.reserve(
            "user",
            "next",
            enabled=True,
            minute_limit=3,
            hour_limit=30,
            day_limit=100,
            now=20000,
        )
        self.assertTrue(allowed, message)

    async def test_filling_quota_blocks_with_daily_message(self):
        ok, _ = await self.store.set_used("group:1", 20, now=5000)
        self.assertTrue(ok)

        allowed, message = await self.store.reserve(
            "group:1",
            "blocked",
            enabled=True,
            minute_limit=3,
            hour_limit=30,
            day_limit=20,
            now=5000,
        )

        self.assertFalse(allowed)
        self.assertIn("滚动24小时限 20 次", message)

    async def test_invalid_counts_are_rejected_without_changes(self):
        await add_successes(self.store, "user", [1000, 1001])

        for value in (-1, RateLimitStore.MAX_MANUAL_COUNT + 1, True, "3", 2.5, None):
            ok, error = await self.store.set_used("user", value, now=1002)
            self.assertFalse(ok, value)
            self.assertIn("整数", error)

        overview, _ = await self.store.overview(now=1002)
        self.assertEqual(overview["user"]["day"], 2)

    async def test_reset_keeps_running_tasks(self):
        await add_successes(self.store, "user", [1000, 1001])
        await self.store.reserve(
            "user",
            "running",
            enabled=True,
            minute_limit=10,
            hour_limit=10,
            day_limit=10,
            now=1002,
        )

        ok, error = await self.store.reset("user", now=1003)

        self.assertTrue(ok, error)
        overview, _ = await self.store.overview(now=1003)
        self.assertEqual(overview["user"]["day"], 0)
        self.assertEqual(overview["user"]["pending"], 1)
        # 重置后完成的任务仍正常记账
        self.assertTrue(
            await self.store.finish("user", "running", successful=True, now=1004)
        )
        overview, _ = await self.store.overview(now=1004)
        self.assertEqual(overview["user"]["day"], 1)

    async def test_reset_all_clears_every_subject_and_persists(self):
        await add_successes(self.store, "user", [1000])
        await add_successes(self.store, "group:1", [1000, 1001])

        ok, error = await self.store.reset_all(now=1002)

        self.assertTrue(ok, error)
        overview, _ = await RateLimitStore(self.data_dir).overview(now=1002)
        self.assertEqual(overview, {})

    async def test_admin_actions_fail_closed_on_corrupt_storage(self):
        (self.data_dir / "rate_limit_usage.json").write_text("{", encoding="utf-8")

        overview, error = await self.store.overview(now=1000)
        self.assertIsNone(overview)
        self.assertIn("限额记录不可用", error)

        ok, error = await self.store.reset("user", now=1000)
        self.assertFalse(ok)
        self.assertIn("限额记录不可用", error)


class SessionRegistryTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._temp_dir.name)
        self.registry = SessionRegistry(self.data_dir)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_describe_subject(self):
        self.assertEqual(describe_subject("group:123"), ("group", "123"))
        self.assertEqual(describe_subject("456"), ("private", "456"))

    def test_touch_keeps_known_names_and_first_seen(self):
        self.registry.touch(
            "group:1",
            name="绘画群",
            platform="aiocqhttp",
            user_id="10",
            user_name="小林",
            now=100,
        )
        self.registry.touch("group:1", name="", user_id="11", user_name="", now=200)

        record = SessionRegistry(self.data_dir).list()["group:1"]
        self.assertEqual(record["kind"], "group")
        self.assertEqual(record["target_id"], "1")
        self.assertEqual(record["name"], "绘画群")
        self.assertEqual(record["platform"], "aiocqhttp")
        self.assertEqual(record["last_user_id"], "11")
        self.assertEqual(record["last_user_name"], "小林")
        self.assertEqual(record["first_seen"], 100)
        self.assertEqual(record["last_seen"], 200)

    def test_success_count_and_remark(self):
        self.registry.record_success("user", now=100)
        self.registry.record_success("user", now=200)
        self.registry.set_remark("user", "  " + "长" * 100)

        record = self.registry.list()["user"]
        self.assertEqual(record["kind"], "private")
        self.assertEqual(record["total_success"], 2)
        self.assertEqual(record["last_success"], 200)
        self.assertEqual(record["remark"], "长" * SessionRegistry.MAX_TEXT_LENGTH)

        self.registry.set_remark("user", "")
        self.assertEqual(self.registry.list()["user"]["remark"], "")

    def test_invalid_subjects_are_rejected(self):
        for subject in ("", "   ", "group:", None, "x" * 300):
            with self.assertRaises(ValueError):
                self.registry.touch(subject)

    def test_corrupt_file_is_not_overwritten(self):
        path = self.data_dir / "sessions.json"
        path.write_text("not json", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.registry.touch("user", now=100)
        self.assertEqual(path.read_text(encoding="utf-8"), "not json")


# ---------- main.py 中的面板接口 ----------


class _NullLogger:
    @staticmethod
    def info(*args, **kwargs):
        pass

    @staticmethod
    def warning(*args, **kwargs):
        pass


class _FakeRequest:
    def __init__(self, payload: Any):
        self.payload = payload

    async def json(self, default=None):
        return self.payload


class _FakeConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved = 0

    def save_config(self):
        self.saved += 1


class _FakeContext:
    def __init__(self):
        self.registered_web_apis = []

    def register_web_api(self, route, handler, methods, desc):
        self.registered_web_apis.append((route, handler, methods, desc))


def _install_web_stub() -> types.ModuleType:
    """提供 astrbot.api.web，让面板接口按 AstrBot v4.26+ 的方式读取请求。"""
    astrbot_module = sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
    api_module = sys.modules.setdefault("astrbot.api", types.ModuleType("astrbot.api"))
    astrbot_module.api = api_module
    web_module = sys.modules.get("astrbot.api.web")
    if web_module is None:
        web_module = types.ModuleType("astrbot.api.web")
        sys.modules["astrbot.api.web"] = web_module
    api_module.web = web_module
    return web_module


PANEL_METHODS = (
    "_positive_int",
    "_as_bool",
    "_get_rate_limit_settings",
    "_check_permission",
    "_get_rate_limit_subject",
    "_register_session",
    "_record_session_success",
    "_register_web_apis",
    "_panel_ok",
    "_panel_error",
    "_read_panel_payload",
    "_read_panel_subject",
    "panel_list_sessions",
    "panel_set_usage",
    "panel_reset_usage",
    "panel_reset_all",
    "panel_set_remark",
    "panel_set_permission",
    "terminate",
)


def _main_module() -> ast.Module:
    return ast.parse(Path("main.py").read_text(encoding="utf-8"))


def _plugin_class(module: ast.Module) -> ast.ClassDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GeminiImagePlugin"
    )


def _plugin_harness():
    methods = [
        copy.deepcopy(node)
        for node in _plugin_class(_main_module()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in PANEL_METHODS
    ]
    for method in methods:
        method.decorator_list = [
            decorator
            for decorator in method.decorator_list
            if isinstance(decorator, ast.Name)
            and decorator.id in {"staticmethod", "classmethod"}
        ]
    harness = ast.ClassDef(
        name="PluginHarness", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    ast.fix_missing_locations(harness)
    namespace = {
        "Any": Any,
        "AstrMessageEvent": object,
        "PLUGIN_NAME": PLUGIN_NAME,
        "SessionRegistry": SessionRegistry,
        "describe_subject": describe_subject,
        "logger": _NullLogger(),
        "time": time,
    }
    exec(
        compile(ast.Module(body=[harness], type_ignores=[]), "<panel>", "exec"),
        namespace,
    )
    return namespace["PluginHarness"]


class PanelApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(self._temp_dir.name)
        self.web = _install_web_stub()
        self.plugin = _plugin_harness()()
        self.plugin.context = _FakeContext()
        self.plugin.rate_limit_store = RateLimitStore(data_dir)
        self.plugin.session_registry = SessionRegistry(data_dir)
        self.plugin.generator = None
        self.plugin.background_tasks = set()
        self.plugin.config = _FakeConfig(
            {
                "generate_config": {
                    "enable_rate_limit": True,
                    "max_requests_per_minute": 3,
                    "max_requests_per_hour": 30,
                    "max_requests_per_day": 100,
                },
                "permission_config": {
                    "mode": "blacklist",
                    "users": ["blocked-user", 42],
                    "groups": ["9"],
                },
            }
        )

    async def asyncTearDown(self):
        self._temp_dir.cleanup()

    async def call(self, handler: str, payload: Any = None):
        self.web.request = _FakeRequest(payload)
        return await getattr(self.plugin, handler)()

    async def test_list_merges_registry_usage_and_permission_lists(self):
        self.plugin.session_registry.touch("group:1", name="绘画群", now=200)
        self.plugin.session_registry.touch("user", name="小明", now=300)
        now = time.time()
        await add_successes(self.plugin.rate_limit_store, "group:1", [now - 10])
        await add_successes(self.plugin.rate_limit_store, "group:5", [now - 20])

        result = await self.call("panel_list_sessions")

        self.assertEqual(result["status"], "ok")
        data = result["data"]
        self.assertEqual(
            data["limits"], {"enabled": True, "minute": 3, "hour": 30, "day": 100}
        )
        self.assertEqual(data["permission_mode"], "blacklist")
        sessions = {item["subject"]: item for item in data["sessions"]}
        self.assertEqual(
            set(sessions),
            {"group:1", "group:5", "group:9", "user", "blocked-user", "42"},
        )
        # 按最近使用时间排序
        self.assertEqual(
            [item["subject"] for item in data["sessions"][:2]], ["user", "group:1"]
        )
        self.assertEqual(sessions["group:1"]["name"], "绘画群")
        self.assertEqual(sessions["group:1"]["usage"]["day"], 1)
        self.assertEqual(sessions["group:5"]["kind"], "group")
        self.assertEqual(sessions["group:5"]["usage"]["day"], 1)
        self.assertTrue(sessions["group:9"]["listed"])
        self.assertFalse(sessions["group:9"]["allowed"])
        self.assertFalse(sessions["42"]["allowed"])
        self.assertTrue(sessions["user"]["allowed"])
        self.assertEqual(sessions["user"]["usage"]["day"], 0)
        json.dumps(result)

    async def test_allowed_follows_whitelist_mode(self):
        self.plugin.config["permission_config"]["mode"] = "whitelist"
        self.plugin.session_registry.touch("other", now=100)

        sessions = {
            item["subject"]: item
            for item in (await self.call("panel_list_sessions"))["data"]["sessions"]
        }

        self.assertTrue(sessions["group:9"]["allowed"])
        self.assertFalse(sessions["other"]["allowed"])

    async def test_registry_failure_is_reported_as_warning(self):
        self.plugin.session_registry.path.write_text("[", encoding="utf-8")

        result = await self.call("panel_list_sessions")

        self.assertEqual(result["status"], "ok")
        self.assertIn("会话记录读取失败", result["data"]["warning"])

    async def test_set_usage_and_reset(self):
        result = await self.call("panel_set_usage", {"subject": "group:1", "used": 7})
        self.assertEqual(result["status"], "ok")
        overview, _ = await self.plugin.rate_limit_store.overview()
        self.assertEqual(overview["group:1"]["day"], 7)

        result = await self.call("panel_reset_usage", {"subject": "group:1"})
        self.assertEqual(result["status"], "ok")
        overview, _ = await self.plugin.rate_limit_store.overview()
        self.assertNotIn("group:1", overview)

    async def test_bad_requests_return_errors(self):
        cases = [
            ("panel_set_usage", {"subject": "user", "used": -1}),
            ("panel_set_usage", {"subject": "user", "used": "5"}),
            ("panel_set_usage", {"used": 5}),
            ("panel_reset_usage", {"subject": "group:"}),
            ("panel_set_remark", {"subject": ""}),
            ("panel_set_permission", {"subject": "user", "listed": "yes"}),
            ("panel_set_usage", ["not", "a", "dict"]),
        ]
        for handler, payload in cases:
            result = await self.call(handler, payload)
            self.assertEqual(result["status"], "error", (handler, payload))
            self.assertTrue(result["message"])
        self.assertEqual(self.plugin.config.saved, 0)

    async def test_quart_request_is_used_before_astrbot_web_api(self):
        class QuartRequest:
            @staticmethod
            async def get_json(silent=False):
                return {"subject": "user", "used": 2}

        quart_module = types.ModuleType("quart")
        quart_module.request = QuartRequest()
        saved = {name: sys.modules.get(name) for name in ("astrbot.api.web", "quart")}
        sys.modules.pop("astrbot.api.web", None)
        sys.modules["quart"] = quart_module
        try:
            result = await self.plugin.panel_set_usage()
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(result["status"], "ok")
        overview, _ = await self.plugin.rate_limit_store.overview()
        self.assertEqual(overview["user"]["day"], 2)

    async def test_reset_all(self):
        await add_successes(self.plugin.rate_limit_store, "a", [time.time()])
        await add_successes(self.plugin.rate_limit_store, "group:b", [time.time()])

        result = await self.call("panel_reset_all", {})

        self.assertEqual(result["status"], "ok")
        overview, _ = await self.plugin.rate_limit_store.overview()
        self.assertEqual(overview, {})

    async def test_set_remark(self):
        result = await self.call(
            "panel_set_remark", {"subject": "group:1", "remark": " 项目群 "}
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            self.plugin.session_registry.list()["group:1"]["remark"], "项目群"
        )

    async def test_permission_toggle_updates_matching_list(self):
        result = await self.call(
            "panel_set_permission", {"subject": "group:1", "listed": True}
        )
        self.assertEqual(result["status"], "ok")
        perm_conf = self.plugin.config["permission_config"]
        self.assertEqual(perm_conf["groups"], ["9", "1"])
        self.assertFalse(self.plugin._check_permission("anyone", "1"))

        await self.call("panel_set_permission", {"subject": "42", "listed": False})
        await self.call("panel_set_permission", {"subject": "group:1", "listed": True})
        self.assertEqual(perm_conf["users"], ["blocked-user"])
        self.assertEqual(perm_conf["groups"], ["9", "1"])
        self.assertTrue(self.plugin._check_permission("42", ""))
        self.assertEqual(self.plugin.config.saved, 3)

    async def test_permission_toggle_creates_missing_config(self):
        del self.plugin.config["permission_config"]

        await self.call("panel_set_permission", {"subject": "user", "listed": True})

        self.assertEqual(self.plugin.config["permission_config"]["users"], ["user"])

    async def test_register_session_from_events(self):
        class FakeEvent:
            def __init__(self, group_name, sender_name):
                self.message_obj = types.SimpleNamespace(
                    group=types.SimpleNamespace(group_name=group_name)
                    if group_name is not None
                    else None
                )
                self.sender_name = sender_name

            def get_sender_name(self):
                return self.sender_name

            @staticmethod
            def get_platform_name():
                return "aiocqhttp"

        self.plugin._register_session(FakeEvent("绘画群", "小林"), "group:1", "10")
        self.plugin._register_session(FakeEvent(None, "小明"), "20", "20")
        # 事件缺少接口时只记录警告，不影响生图
        self.plugin._register_session(object(), "30", "30")
        self.plugin._record_session_success("group:1")

        records = self.plugin.session_registry.list()
        self.assertEqual(records["group:1"]["name"], "绘画群")
        self.assertEqual(records["group:1"]["last_user_name"], "小林")
        self.assertEqual(records["group:1"]["total_success"], 1)
        self.assertEqual(records["20"]["name"], "小明")
        self.assertNotIn("30", records)

    async def test_routes_match_page_and_are_removed_on_terminate(self):
        self.plugin._register_web_apis()
        other_handler = object()
        self.plugin.context.registered_web_apis.append(
            ("/other_plugin/x", other_handler, ["GET"], "")
        )

        routes = {
            (route, tuple(methods))
            for route, _, methods, _ in self.plugin.context.registered_web_apis
        }
        page_calls = re.findall(
            r'bridge\.api(Get|Post)\("([^"]+)"',
            Path("pages/sessions/app.js").read_text(encoding="utf-8"),
        )
        self.assertTrue(page_calls)
        for method, endpoint in page_calls:
            self.assertIn((f"/{PLUGIN_NAME}/{endpoint}", (method.upper(),)), routes)

        await self.plugin.terminate()

        self.assertEqual(
            self.plugin.context.registered_web_apis,
            [("/other_plugin/x", other_handler, ["GET"], "")],
        )


class EntrypointTests(unittest.TestCase):
    def test_entrypoints_register_session_before_reserving_quota(self):
        module = _main_module()
        tool_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "GeminiImageGenerationTool"
        )
        entrypoints = [
            next(
                node
                for node in tool_class.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "call"
            ),
            next(
                node
                for node in _plugin_class(module).body
                if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "generate_image_command"
            ),
        ]
        for entrypoint in entrypoints:
            calls = [
                node
                for node in ast.walk(entrypoint)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ]
            lines = {}
            for node in calls:
                lines.setdefault(node.func.attr, node.lineno)
            self.assertLess(lines["_register_session"], lines["_reserve_rate_limit"])
            background = next(
                node
                for node in calls
                if node.func.attr == "_generate_and_send_image_async"
            )
            self.assertIn(
                "session_subject_id", {keyword.arg for keyword in background.keywords}
            )

    def test_success_path_counts_session(self):
        method = next(
            node
            for node in _plugin_class(_main_module()).body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_generate_and_send_image_async"
        )
        called = {
            node.func.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_record_session_success", called)

    def test_page_files_and_i18n_exist(self):
        page_dir = Path("pages/sessions")
        html = (page_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn('src="./app.js"', html)
        self.assertIn('href="./style.css"', html)
        for locale in ("zh-CN", "en-US"):
            data = json.loads(
                Path(f".astrbot-plugin/i18n/{locale}.json").read_text(encoding="utf-8")
            )
            self.assertTrue(data["pages"]["sessions"]["title"])


if __name__ == "__main__":
    unittest.main()
