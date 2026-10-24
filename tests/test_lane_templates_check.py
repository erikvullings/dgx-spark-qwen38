#!/usr/bin/env python3
"""tests/lane-templates-check.py, without the network: the text lanes it knows are every text lane,
each served the way its installer serves it, and the shapes it renders are the captured ones.

The check itself, with the network and jinja2, renders every lane's template at its pin (a CI step of
its own): the Ornith lane's published template refused opencode 1.19.4's second system message, a 400
on every opencode request from v1.38.6 to v1.40.1 that no test saw (2026-10-24)."""
import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lane_templates_check", REPO / "tests" / "lane-templates-check.py")
ltc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ltc)


class TheLanesItKnows(unittest.TestCase):
    def test_every_text_lane_with_how_its_installer_serves_its_template(self):
        specs = {name: env for name, _, _, env in ltc.lane_specs()}
        self.assertEqual(sorted(specs), sorted(["stock", "uncensored", "fp8", "uncensored-fp8", "flash", "flash-nvda",
                                                "flash-uncensored", "flash-local", "ornith-122b", "kolibri",
                                                "kolibri-nvfp4", "nemotron", "nemotron-super"]))
        self.assertEqual(specs["ornith-122b"], {"patches": "system-reminder"})
        self.assertEqual(specs["stock"], {})
        self.assertIsNone(specs["kolibri"])

    def test_each_kolibri_checkpoint_at_its_own_pin(self):
        # the check read only the NVFP4's pin, while the FP8 is the lane's default (2026-10-24)
        repos = {name: repo for name, repo, _, _ in ltc.lane_specs()}
        self.assertEqual(repos["kolibri"], "Aleph-Alpha/Kolibri-1")
        self.assertEqual(repos["kolibri-nvfp4"], "Quantinine/Kolibri-1-NVFP4A16-FP8")

    def copy(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="ltc-"))
        self.addCleanup(shutil.rmtree, d, True)
        for f in [REPO / "install.sh", REPO / "switch-model.sh", *REPO.glob("install-*.sh")]:
            shutil.copy(f, d / f.name)
        return d

    def test_a_switch_target_it_does_not_know_stops_it(self):
        d = self.copy()
        f = d / "switch-model.sh"
        f.write_text(f.read_text().replace('case "$CHOICE" in stock|', 'case "$CHOICE" in stock|newlane|', 1))
        with self.assertRaisesRegex(SystemExit, r"text targets \['newlane'\] this check does not know"):
            ltc.lane_specs(d)

    def test_a_vllm_lane_it_does_not_know_stops_it(self):
        d = self.copy()
        (d / "install-newlane.sh").write_text('vlane_put_unit "$RENDER"\n')
        with self.assertRaisesRegex(SystemExit, "add the new lane to VLLM_LANES"):
            ltc.lane_specs(d)

    def test_an_installer_that_serves_its_template_otherwise_stops_it(self):
        d = self.copy()
        f = d / "install-ornith-122b.sh"
        f.write_text(f.read_text().replace("TEMPLATE_FIXES=(--patches=system-reminder)", "TEMPLATE_FIXES=()"))
        with self.assertRaisesRegex(SystemExit, "install-ornith-122b.sh serves its template otherwise"):
            ltc.lane_specs(d)


class WhatItRendersAsTheEnginesDo(unittest.TestCase):
    # three ways the check rendered otherwise than the engines and the installers (code review, 2026-10-24)
    def test_named_templates_give_tool_use_when_tools_are_sent(self):
        named = {"default": "D", "tool_use": "T"}
        self.assertEqual((ltc.pick(named, None), ltc.pick(named, ltc.TOOLS)), ("D", "T"))
        self.assertEqual(ltc.pick({"default": "D"}, ltc.TOOLS), "D")
        self.assertEqual(ltc.pick("one", ltc.TOOLS), "one")

    def test_each_content_also_goes_as_a_list_of_parts(self):
        m = ltc.messages(["system", "user", "assistant_tool_call", "tool"], parts=True)
        self.assertEqual(m[1]["content"], [{"type": "text", "text": "A user message, number 1."}])
        self.assertEqual(m[3]["content"], [{"type": "text", "text": "7351"}])
        env = ltc.environment()
        strings_only = "{% for m in messages %}{{ m.content + '!' }}{% endfor %}"
        self.assertIsNone(ltc.render(env, strings_only, ["user"], None))
        self.assertIsNotNone(ltc.render(env, strings_only, ["user"], None, parts=True))

    def test_a_patched_lane_needs_the_file_its_installer_patches(self):
        with self.assertRaisesRegex(SystemExit, "its installer patches chat_template.jinja"):
            ltc.served("org/x", "a" * 40, ("tpl", "tokenizer_config.json"), {"patches": "system-reminder"})
        self.assertEqual(ltc.served("org/x", "a" * 40, ("tpl", "tokenizer_config.json"), None), "tpl")

    def test_a_429_is_waited_out_and_a_404_is_none(self):
        import io
        import urllib.error
        calls, slept = [], []

        def urlopen(url, timeout=60):
            calls.append(url)
            if url.endswith("chat_template.jinja") and len(calls) < 3:
                raise urllib.error.HTTPError(url, 429, "busy", {"Retry-After": "7"}, None)
            if url.endswith("chat_template.jinja"):
                return io.BytesIO(b"TEMPLATE")
            raise urllib.error.HTTPError(url, 404, "no", {}, None)
        real = ltc.urllib.request.urlopen
        ltc.urllib.request.urlopen = urlopen
        try:
            self.assertEqual(ltc.fetch("org/x", "a" * 40, sleep=slept.append), ("TEMPLATE", "chat_template.jinja"))
        finally:
            ltc.urllib.request.urlopen = real
        self.assertEqual((len(calls), slept), (3, [7, 7]))


class TheContentFormEachEngineHands(unittest.TestCase):
    # vLLM turns a string into a list of parts for a template that loops over a message's content,
    # SGLang leaves strings alone: rendering lists everywhere refused Kolibri's templates, which
    # neither engine hands a list (2026-10-24)
    def setUp(self):
        self.env = ltc.environment()

    def test_vllm_hands_parts_to_a_template_that_loops_over_the_content(self):
        loops = "{% for m in messages %}{% for c in m['content'] %}{{ c.text }}{% endfor %}{% endfor %}"
        concat = "{% for m in messages %}{{ m['content'] + '!' }}{% endfor %}"
        self.assertEqual(ltc.content_forms(loops, "vllm", self.env), [True])
        self.assertEqual(ltc.content_forms(concat, "vllm", self.env), [False])

    def test_vllm_follows_the_content_into_a_macro(self):
        macro = ("{% macro show(body) %}{% for part in body %}{{ part.text }}{% endfor %}{% endmacro %}"
                 "{% for m in messages %}{{ show(m.content) }}{% endfor %}")
        self.assertEqual(ltc.content_forms(macro, "vllm", self.env), [True])

    def test_sglang_keeps_strings_and_takes_lists_only_where_the_template_does(self):
        loops = "{% for message in messages %}{% for c in message.content %}{{ c.text }}{% endfor %}{% endfor %}"
        self.assertEqual(ltc.content_forms(loops, "sglang", self.env), [False, True])
        self.assertEqual(ltc.content_forms("{{ messages[0].content }}", "sglang", self.env), [False])
        self.assertEqual(ltc.content_forms("{# image #}{{ messages[0].content }}", "sglang", self.env), [False, True])


class TheShapesItRenders(unittest.TestCase):
    def test_opencodes_two_system_messages_are_there(self):
        shapes = json.loads(ltc.SHAPES.read_text())["shapes"]
        self.assertIn(["system", "system", "user"], [s["roles"] for s in shapes])

    def test_a_tool_round_is_a_call_and_its_answer(self):
        m = ltc.messages(["system", "user", "assistant_tool_call", "tool"])
        self.assertEqual([x["role"] for x in m], ["system", "user", "assistant", "tool"])
        self.assertEqual(m[3]["tool_call_id"], m[2]["tool_calls"][0]["id"])
        self.assertIsInstance(m[2]["tool_calls"][0]["function"]["arguments"], dict)


if __name__ == "__main__":
    unittest.main()
