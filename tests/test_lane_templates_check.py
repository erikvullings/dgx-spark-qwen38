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
                                                "nemotron", "nemotron-super"]))
        self.assertEqual(specs["ornith-122b"], {"TEMPLATE_PATCHES": "system-reminder"})
        self.assertEqual(specs["stock"], {})
        self.assertIsNone(specs["kolibri"])

    def copy(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="ltc-"))
        self.addCleanup(shutil.rmtree, d, True)
        for f in [REPO / "install.sh", *REPO.glob("install-*.sh")]:
            shutil.copy(f, d / f.name)
        return d

    def test_a_vllm_lane_it_does_not_know_stops_it(self):
        d = self.copy()
        (d / "install-newlane.sh").write_text('vlane_put_unit "$RENDER"\n')
        with self.assertRaisesRegex(SystemExit, "add the new lane to VLLM_LANES"):
            ltc.lane_specs(d)

    def test_an_installer_that_serves_its_template_otherwise_stops_it(self):
        d = self.copy()
        f = d / "install-ornith-122b.sh"
        f.write_text(f.read_text().replace("TEMPLATE_PATCHES=system-reminder ", ""))
        with self.assertRaisesRegex(SystemExit, "install-ornith-122b.sh serves its template otherwise"):
            ltc.lane_specs(d)


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
