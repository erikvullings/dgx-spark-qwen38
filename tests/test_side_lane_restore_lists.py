#!/usr/bin/env python3
"""The image and video installers name, in lists of their own, the engines their lane's start stops.

Their proof starts their lane, whose unit's Conflicts= stops whatever engine served, and starts that
engine again after; their uninstall gives the boot back when no engine is enabled. Each reads the
engines from a hand-kept list: the proof's must be the unit's Conflicts= set, and the Flash FP4-Local
lane, new in Conflicts=, was in neither list, so a proof run while it served left the box serving
nothing (found in review, 2026-10-20). The lists are read from the scripts as written."""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
LIST = re.compile(r"^\s*for u in ((?:qwen38-[\w-]+\.service ?)+); do\n\s*systemctl is-(active|enabled) ", re.M)


def lists(lane):
    return {kind: set(units.split()) for units, kind in LIST.findall((REPO / f"install-{lane}.sh").read_text())}


def conflicts(lane):
    text = (REPO / f"qwen38-{lane}.service.template").read_text()
    return set(re.search(r"^Conflicts=(.*)$", text, re.M).group(1).split())


class TheListsFollowTheUnits(unittest.TestCase):
    def test_the_proof_starts_again_any_engine_the_lanes_start_stops(self):
        for lane in ("image", "video"):
            with self.subTest(lane=lane):
                self.assertEqual(lists(lane)["active"], conflicts(lane))

    def test_the_uninstall_sees_every_vllm_lane_at_boot(self):
        for lane in ("image", "video"):
            with self.subTest(lane=lane):
                enabled = lists(lane)["enabled"]
                for unit in ("qwen38-kolibri.service", "qwen38-nemotron.service",
                             "qwen38-nemotron-super.service", "qwen38-flash-local.service"):
                    self.assertIn(unit, enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
