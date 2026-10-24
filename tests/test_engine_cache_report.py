#!/usr/bin/env python3
"""The 27B's units and run.sh start SGLang with --enable-cache-report (discussion #49, 2026-10-24).

With it, each answer says how much of its prompt the prefix cache served:
usage.prompt_tokens_details.cached_tokens on the OpenAI routes, cache_read_input_tokens beside
input_tokens on /v1/messages, and the proxy's x-systemone-cached-tokens. Without it the cache
serves the same and says nothing. Measured on the 27B lane behind the proxy on 2026-10-24, with the
flag and without: the SDKs, fetch under Node and Bun, LiteLLM, Claude Code, opencode, Hermes and pi
each did their task, opencode's total and Claude Code's count stayed what they were, and bench.sh
gave 64.9 against 64.0 tok/s (three runs each, their spreads overlapping). The recipes' drift check
knows the switch, so a 27B unit written before it says so in the cockpit.
"""
import pathlib
import re
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
LAUNCHES = ("qwen38-sglang.service.template", "qwen38-sglang-1m.service.template",
            "qwen38-sglang-64.service.template", "run.sh")
FLAG = re.compile(r"(?<![\w-])--enable-cache-report(?![\w-])")


def code(name):
    """The file without its comment lines: a flag named in prose is not passed."""
    return "\n".join(ln for ln in (REPO / name).read_text().splitlines() if not ln.lstrip().startswith("#"))


class TheFlagIsPassed(unittest.TestCase):
    def test_each_27b_launch_passes_it_once(self):
        for name in LAUNCHES:
            with self.subTest(name=name):
                self.assertEqual(len(FLAG.findall(code(name))), 1)

    def test_the_recipes_see_a_unit_without_it(self):
        sys.path.insert(0, str(REPO / "dashboard"))
        import recipes
        self.assertIn("--enable-cache-report", recipes.SWITCHES)
        self.assertTrue(recipes._switches("--enable-metrics --enable-cache-report")["--enable-cache-report"])
        self.assertFalse(recipes._switches("--enable-metrics --enable-cache-reports")["--enable-cache-report"])


if __name__ == "__main__":
    unittest.main()
