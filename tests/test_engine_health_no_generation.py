#!/usr/bin/env python3
"""GET /health generates nothing on the SGLang text lanes (2026-10-24).

In both pinned SGLang images (the 27B's v0.5.22 and the flash lane's nightly), GET /health is
/health_generate: on an idle engine it runs a one-token generation, unless
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION is false (srt/environ.py, EnvBool(True);
srt/entrypoints/http_server.py, health_generate). Read in both images on 2026-10-24: "0" parses
as false, and a server that is still starting answers 503 before that test, so /health stays the
readiness the installers and the cockpit wait on. /health takes no key, and the proxy relayed it
from the network: anyone who reached the proxy made the engine generate (D02 of the 2026-10-24
security review). Every SGLang engine start of this repo passes the variable at 0 now, and the
proxy asks the key for /health_generate (tests/test_proxy_door.py).
"""
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAG = "-e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 "


def sglang_starts():
    """Every file of the repo that starts an SGLang text engine."""
    return sorted(p.name for p in [*REPO.glob("*.template"), REPO / "run.sh"]
                  if "sglang.launch_server" in p.read_text())


class EveryEngineStartTurnsTheHealthGenerationOff(unittest.TestCase):
    def test_the_starts_are_the_ones_this_test_knows(self):
        # a new SGLang lane is held to it as soon as its unit exists
        self.assertEqual(sglang_starts(), ["qwen38-flash-launch.sh.template", "qwen38-sglang-1m.service.template",
                                           "qwen38-sglang-64.service.template", "qwen38-sglang.service.template",
                                           "run.sh"])

    def test_each_docker_run_passes_it_once(self):
        for name in sglang_starts():
            with self.subTest(name=name):
                text = (REPO / name).read_text()
                run = text[text.index("docker run"):]
                self.assertEqual(run.count(FLAG), 1)

    def test_the_units_say_why(self):
        for name in sglang_starts():
            if name == "run.sh":
                continue                      # verbatim the native unit's docker run (CI checks it)
            with self.subTest(name=name):
                text = (REPO / name).read_text()
                self.assertIn("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0: GET /health answers from the server's state",
                              text[:text.index("docker run")])


if __name__ == "__main__":
    unittest.main()
