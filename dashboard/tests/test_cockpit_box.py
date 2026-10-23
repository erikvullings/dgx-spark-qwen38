#!/usr/bin/env python3
"""The machine panel says which box this is: its memory class and the name its maker sells it
under (boxmem.py), and the lanes' units that fit it, which the page reads to say a lane a 64 GB
DGX Spark cannot hold needs a 128 GB one (2026-10-23). The cockpit runs dry here, on a DMI of the
test's own."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(DASH))
ENV_BEFORE = dict(os.environ)


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)


class TheBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-box-"))
        (cls.tmp / "config").mkdir()
        (cls.tmp / "config" / "api-key").write_text("k\n")
        dmi = cls.tmp / "dmi"
        dmi.mkdir()
        for k, v in (("sys_vendor", "ASUSTeK COMPUTER INC."), ("product_name", "GX10"), ("product_family", "DGX Spark")):
            (dmi / k).write_text(v + "\n")
        closed = "http://127.0.0.1:1"
        os.environ.update(HOME=str(cls.tmp), COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed, COCKPIT_IMAGE=closed,
                          QWEN38_DMI_DIR=str(dmi))
        spec = importlib.util.spec_from_file_location("cockpit_box_under_test", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def box(self, cls):
        os.environ["QWEN38_MEMORY_CLASS"] = cls
        self.addCleanup(os.environ.pop, "QWEN38_MEMORY_CLASS", None)
        self.cp._BOX.clear()
        self.addCleanup(self.cp._BOX.clear)
        return self.cp.box_info()

    def test_a_64_gb_box_and_the_two_lanes_that_fit_it(self):
        self.assertEqual(self.box("64"), {"cls": "64", "name": "ASUS Ascent GX10",
                                          "fits": ["qwen38-nemotron.service", "qwen38-sglang.service"]})

    def test_a_128_gb_box_and_every_lane(self):
        got = self.box("128")
        self.assertEqual((got["cls"], got["name"]), ("128", "ASUS Ascent GX10"))
        self.assertEqual(set(got["fits"]), set(self.cp.UNIT_OF_LANE.values()))

    def test_the_machine_panel_carries_it(self):
        self.box("64")
        self.assertEqual(self.cp.collect_machine()["box"]["cls"], "64")

    def test_a_box_that_cannot_be_read_is_unknown_not_a_failed_panel(self):
        os.environ["QWEN38_MEMORY_CLASS"] = "96"      # boxmem.py refuses it
        self.addCleanup(os.environ.pop, "QWEN38_MEMORY_CLASS", None)
        self.cp._BOX.clear()
        self.addCleanup(self.cp._BOX.clear)
        self.assertEqual(self.cp.box_info(), {"cls": None, "name": None, "fits": None})
        self.assertIn("mem", self.cp.collect_machine())


if __name__ == "__main__":
    unittest.main()
