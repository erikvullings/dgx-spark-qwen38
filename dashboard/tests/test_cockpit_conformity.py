"""The cockpit reads this box against what its clone recommends (v1.36.0, 2026-10-17).

The list a run of install.sh prints before its first step said, once per update, what the box
kept and what it moved; between updates nothing said it, and an operator who never read an
update's output never saw a value an older release had left behind. The update tier now carries
conformity.py's plan for the text lane the box serves from (`conformity`, the same list, with
nothing set on a command line), read again only once a file it reads has changed, and nothing at
all when conformity.py cannot say. The repo's modules the cockpit loads once are watched for a
change too, here and by the installer that restarts it: conformity.py was loaded and watched by
neither.
"""
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = DASH.parent
sys.path.insert(0, str(REPO / "tests"))
import test_conformity_classify as tcc  # noqa: E402

ENV_BEFORE = {}
OLD_BOX = "2026-09-05T09:00:00+02:00"


def setUpModule():
    ENV_BEFORE.update(os.environ)
    # conformity.py recommends by the box's class (the native window on a 64 GB one): the cockpit
    # here holds a 128 GB box's state, whatever the machine running it is, and an exported class is
    # back as it was after (tearDownModule; found in review, 2026-10-23)
    os.environ["QWEN38_MEMORY_CLASS"] = "128"


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


def cockpit(units, cfg, name):
    """The cockpit, loaded on a box whose units and config folder are these, its user's home the
    one the units were rendered for (the recommended cache is under it)."""
    os.environ.update(HOME=tcc.HOME, COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cfg), COCKPIT_UNIT_DIR=str(units),
                      COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                      COCKPIT_AUTOHEAL="0", COCKPIT_UPDATE_CHECK="0")
    sys.path.insert(0, str(DASH))
    spec = importlib.util.spec_from_file_location(name, DASH / "cockpit.py")
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    return cp


class TheUpdateTierCarriesIt(unittest.TestCase):
    def setUp(self):
        self.home_was = tcc.HOME
        self.home = Path(tempfile.mkdtemp(prefix="cockpit-conform-"))
        self.addCleanup(shutil.rmtree, self.home, True)
        tcc.HOME = str(self.home)

    def tearDown(self):
        tcc.HOME = self.home_was

    def box(self, name, **kw):
        units, cfg = tcc.box(**kw)
        return cockpit(units, cfg, name), Path(units), Path(cfg)

    def test_a_box_as_recommended_carries_an_empty_list(self):
        cp, _, _ = self.box("cockpit_conform_a", key_at=OLD_BOX)
        got = cp.conformity_state()
        self.assertEqual(got["items"], [])
        self.assertEqual(got["lane"], "27b")
        self.assertEqual(got["release"], re.search(r"^## (v\d+\.\d+(?:\.\d+)?)\b",
                                                   (REPO / "CHANGELOG.md").read_text(), re.M).group(1))

    def test_an_old_default_and_a_site_setting_are_listed_as_the_run_lists_them(self):
        cp, units, cfg = self.box("cockpit_conform_b", host="0.0.0.0", key_at=OLD_BOX, hf="/mnt/ssd/huggingface")
        items = cp.conformity_state()["items"]
        self.assertEqual([(i["group"], i["knob"], i["value"]) for i in items],
                         [("moves", "ENGINE_BIND", "0.0.0.0"), ("site", "HF_CACHE", "/mnt/ssd/huggingface")])
        conformity = cp._conformity()
        self.assertEqual(items, conformity.plan(cfg, "27b", {}, units_dir=units, home=tcc.HOME),
                         "the run's own list, nothing set on a command line")

    def test_the_lane_is_the_one_the_record_names(self):
        rec = tcc.a_record()
        rec["lane"] = "flash"
        cp, _, _ = self.box("cockpit_conform_c", record=rec)
        self.assertEqual(cp.conformity_state()["lane"], "flash")

    def test_it_is_read_again_only_once_a_file_it_reads_has_changed(self):
        cp, units, cfg = self.box("cockpit_conform_d", key_at=OLD_BOX)
        mod = cp._conformity()
        calls = []
        real = mod.plan
        mod.plan = lambda *a, **k: calls.append(1) or real(*a, **k)
        self.addCleanup(setattr, mod, "plan", real)
        cp.conformity_state()
        cp.conformity_state()
        self.assertEqual(len(calls), 1, "the second read changed nothing")
        unit = units / "qwen38-sglang.service"
        unit.write_text(unit.read_text().replace("--host 127.0.0.1", "--host 0.0.0.0"))
        later = time.time() + 5
        os.utime(unit, (later, later))
        got = cp.conformity_state()
        self.assertEqual(len(calls), 2)
        self.assertEqual([i["knob"] for i in got["items"]], ["ENGINE_BIND"])

    def test_a_side_lanes_switch_is_read_again(self):
        # ./switch-model.sh kolibri-nvfp4 rewrites the Kolibri unit: its checkpoint is listed
        # since 2026-10-17, so its unit is one of the files the cache follows
        cp, units, cfg = self.box("cockpit_conform_side", key_at=OLD_BOX)
        self.assertEqual(cp.conformity_state()["items"], [])
        unit = units / "qwen38-kolibri.service"
        unit.write_text(unit.read_text().replace("Aleph-Alpha/Kolibri-1 --revision",
                                                 "Quantinine/Kolibri-1-NVFP4A16-FP8 --revision"))
        later = time.time() + 5
        os.utime(unit, (later, later))
        got = cp.conformity_state()
        self.assertEqual([(i["knob"], i["value"], i["group"]) for i in got["items"]],
                         [("KOLIBRI_VARIANT", "nvfp4", "yours")])

    def test_what_cannot_be_read_is_not_said(self):
        cp, _, _ = self.box("cockpit_conform_e", key_at=OLD_BOX)
        mod = cp._conformity()
        real = mod.plan
        mod.plan = lambda *a, **k: (_ for _ in ()).throw(KeyError("SECOND_TARGET of the flash lane"))
        self.addCleanup(setattr, mod, "plan", real)
        cp._CONFORM_CACHE.clear()
        self.assertIsNone(cp.conformity_state())
        out = cp.collect_update()
        self.assertNotIn("conformity", out)
        self.assertIn("installed", out, "the rest of the tier answers")

    def test_collect_update_hands_it_to_the_page(self):
        cp, _, _ = self.box("cockpit_conform_f", host="0.0.0.0", key_at=OLD_BOX)
        out = cp.collect_update()
        self.assertEqual([i["knob"] for i in out["conformity"]["items"]], ["ENGINE_BIND"])
        json.dumps(out)


class EveryModuleItLoadsIsWatched(unittest.TestCase):
    def test_each_repo_module_loaded_once_is_in_both_lists(self):
        text = (DASH / "cockpit.py").read_text()
        loaded = set(re.findall(r'spec_from_file_location\("[^"]+", REPO_DIR / "([^"]+\.py)"\)', text))
        self.assertGreaterEqual(loaded, {"engine-inputs.py", "oc-fit-limits.py", "oc-merge-limits.py", "conformity.py"})
        files = re.search(r"^REPO_CODE_FILES = \(([^)]*)\)", text, re.M).group(1)
        self.assertEqual(loaded, set(re.findall(r'"([^"]+)"', files)), "what the stale-code banner watches")
        restart = next(ln for ln in (DASH / "install-dashboard.sh").read_text().splitlines()
                       if 'stale_since "$UNIT"' in ln)
        for name in sorted(loaded):
            with self.subTest(module=name):
                self.assertIn(f'"$REPO_DIR/{name}"', restart, "what the installer restarts the cockpit for")


if __name__ == "__main__":
    unittest.main(verbosity=2)
