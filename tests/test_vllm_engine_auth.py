#!/usr/bin/env python3
"""vllm_engine_auth.RequireKey: the key on every route of the Kolibri lane's engine.

vLLM 0.29 checks --api-key only under /v1, /v2, /inference and /cohere: /invocations
(SageMaker's router, registered on every server, which runs chat and completions),
/tokenize and /detokenize answered anyone who reached the port. The middleware refuses
everything but /health and /metrics without the key, as the SGLang lanes do; the unit
loads it and the installer puts it where the unit mounts it.
"""
import asyncio
import importlib.util
import pathlib
import re
import tempfile
import unittest
import unittest.mock

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("vllm_engine_auth", REPO / "vllm_engine_auth.py")
auth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auth)
KEY = "sk-test-0123456789abcdef"


class Fake:
    """An ASGI app that records being reached, and a send() that records the reply."""
    def __init__(self):
        self.reached, self.sent = [], []

    async def app(self, scope, receive, send):
        self.reached.append(scope["path"])

    async def send(self, msg):
        self.sent.append(msg)


class TheEngineAsksTheKeyEverywhere(unittest.TestCase):
    def setUp(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="vllm-auth-"))
        self.keyfile = d / "api-key"
        self.keyfile.write_text(KEY + "\n")
        self.env = unittest.mock.patch.dict("os.environ", {"QWEN38_KEY_FILE": str(self.keyfile)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def call(self, path, auth_header=None, kind="http", method="POST", root=""):
        fake = Fake()
        mw = auth.RequireKey(fake.app)
        headers = [(b"content-type", b"application/json")]
        if auth_header is not None:
            headers.append((b"authorization", auth_header.encode()))
        scope = {"type": kind, "path": path, "root_path": root, "method": method, "headers": headers}
        asyncio.run(mw(scope, None, fake.send))
        status = next((m["status"] for m in fake.sent if m["type"] == "http.response.start"), None)
        return fake, status

    def test_invocations_without_the_key_is_refused(self):
        fake, status = self.call("/invocations")
        self.assertEqual(status, 401)
        self.assertEqual(fake.reached, [], "the engine never saw it")

    def test_every_unguarded_vllm_route_is_refused_without_the_key(self):
        for path in ("/invocations", "/tokenize", "/detokenize", "/scale_elastic_ep", "/ping",
                     "/version", "/load", "/v1/chat/completions", "/v1/models", "/start_profile"):
            with self.subTest(path=path):
                fake, status = self.call(path)
                self.assertEqual((status, fake.reached), (401, []))

    def test_the_key_opens_them(self):
        for path in ("/invocations", "/tokenize", "/v1/chat/completions", "/version"):
            with self.subTest(path=path):
                fake, status = self.call(path, f"Bearer {KEY}")
                self.assertEqual((status, fake.reached), (None, [path]))

    def test_a_wrong_key_or_another_scheme_is_refused(self):
        for header in (f"Bearer {KEY}x", "Bearer ", f"Basic {KEY}", KEY, f"bearer{KEY}"):
            with self.subTest(header=header):
                fake, status = self.call("/invocations", header)
                self.assertEqual((status, fake.reached), (401, []))

    def test_the_scheme_is_case_blind_as_in_vllm(self):
        fake, status = self.call("/invocations", f"bearer {KEY}")
        self.assertEqual(fake.reached, ["/invocations"])

    def test_health_and_metrics_stay_open(self):
        # the cockpit reads both without the key, on every lane
        for path in ("/health", "/metrics", "/metrics/"):
            with self.subTest(path=path):
                fake, status = self.call(path, method="GET")
                self.assertEqual((status, fake.reached), (None, [path]))

    def test_a_root_path_is_taken_off_first(self):
        fake, status = self.call("/engine/health", root="/engine", method="GET")
        self.assertEqual(fake.reached, ["/engine/health"])
        fake, status = self.call("/engine/invocations", root="/engine")
        self.assertEqual(status, 401)

    def test_preflight_and_lifespan_pass(self):
        fake, status = self.call("/invocations", method="OPTIONS")
        self.assertEqual(fake.reached, ["/invocations"])
        fake = Fake()
        asyncio.run(auth.RequireKey(fake.app)({"type": "lifespan", "path": ""}, None, fake.send))
        self.assertEqual(fake.reached, [""])

    def test_a_websocket_without_the_key_is_closed(self):
        fake, _ = self.call("/v1/realtime", kind="websocket")
        self.assertEqual(fake.reached, [])
        self.assertEqual(fake.sent, [{"type": "websocket.close", "code": 1008}])

    def test_an_empty_key_file_keeps_the_engine_down(self):
        self.keyfile.write_text("\n")
        with self.assertRaises(RuntimeError):
            auth.RequireKey(Fake().app)


class TheLaneLoadsIt(unittest.TestCase):
    def test_the_unit_loads_the_middleware_from_its_own_mount(self):
        unit = (REPO / "qwen38-kolibri.service.template").read_text()
        self.assertIn("--middleware vllm_engine_auth.RequireKey", unit)
        mount = re.search(r"-v (\S+):(/opt/qwen38-auth):ro", unit)
        self.assertIsNotNone(mount, "the module's directory, read-only, and nothing else of the config")
        self.assertIn("-e PYTHONPATH=/opt/qwen38-auth", unit)
        self.assertIn("-e QWEN38_KEY_FILE=/out/api-key", unit)
        self.assertIn("-v __HOME__/.config/qwen38:/out", unit, "where the key file is")

    def test_the_installer_puts_it_there(self):
        inst = (REPO / "install-kolibri.sh").read_text()
        self.assertIn('"$HERE/vllm_engine_auth.py"', inst)
        self.assertIn("$CONFIG_DIR/vllm-auth/vllm_engine_auth.py", inst)


if __name__ == "__main__":
    unittest.main()
