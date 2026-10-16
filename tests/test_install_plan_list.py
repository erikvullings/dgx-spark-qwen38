#!/usr/bin/env python3
"""Before its first step, a run lists this box against what the release recommends (2026-10-17).

An update said each setting as it decided it, one line among a hundred, and said nothing of the
settings it kept quietly: a box could run an old default for good, and its operator never saw it
named beside what a fresh box gets (v1.36.0, "nobody behind in silence"). The main run now
prints one list before step 1: what moves now (an old default nothing records as chosen), what
stays the operator's (recorded, set on this command line, never a default), and the site's own
settings apart; one line when nothing differs. conformity.py classes them by the rules the
convergence asks verdict() for. The run for the other text lane and a --no-service run print
none.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import installer_wall as wall  # noqa: E402
import test_conformity_classify as tcc  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import conformity  # noqa: E402

TEXT = (REPO / "install.sh").read_text()
HOME = tcc.HOME
REC = conformity.recommended(home=HOME)
OLD_BOX = "2026-09-05T09:00:00+02:00"      # before v1.17.0 moved the engine bind
YOUNG_BOX = "2026-10-01T10:00:00+02:00"    # after every change OLD_DEFAULTS lists


def plan(units, cfg, lane="27b", **chose):
    return conformity.plan(cfg, lane, chose, units_dir=units, home=HOME)


def groups(items):
    return {i["knob"]: (i["group"], i["why"]) for i in items}


class WhatTheListSays(unittest.TestCase):
    def test_a_box_as_recommended_says_so_in_one_line(self):
        units, cfg = tcc.box()
        self.assertEqual(plan(units, cfg), [])
        self.assertEqual(conformity.plan_lines([], "v1.36.0"),
                         ["This box against what v1.36.0 recommends: the same, nothing of yours differs."])

    def test_an_old_default_on_an_old_box_moves_now_with_the_variable_that_keeps_it(self):
        units, cfg = tcc.box(host="0.0.0.0", key_at=OLD_BOX)
        items = plan(units, cfg)
        self.assertEqual(groups(items), {"ENGINE_BIND": ("moves", "box-before")})
        line = conformity.plan_lines(items, "v1.36.0")[1]
        self.assertIn("moves now", line)
        self.assertIn("ENGINE_BIND 0.0.0.0, to 127.0.0.1: the default since v1.17.0", line)
        self.assertIn("to keep it: ENGINE_BIND=0.0.0.0 ./install.sh", line)

    def test_the_same_value_on_a_young_box_stays_and_says_why(self):
        units, cfg = tcc.box(host="0.0.0.0", key_at=YOUNG_BOX)
        items = plan(units, cfg)
        self.assertEqual(groups(items), {"ENGINE_BIND": ("yours", "box-after")})
        self.assertIn("this box was installed after v1.17.0 moved the default, so it was chosen",
                      conformity.plan_lines(items, "v1.36.0")[1])

    def test_a_recorded_choice_stays_yours(self):
        units, cfg = tcc.box(tier="throughput", record=tcc.a_record(FLASH_TIER="throughput"))
        items = plan(units, cfg)
        self.assertEqual(groups(items), {"FLASH_TIER": ("yours", "recorded")})
        self.assertIn("FLASH_TIER throughput (recommended: concurrency), recorded as your choice",
                      conformity.plan_lines(items, "v1.36.0")[1])

    def test_a_site_setting_is_listed_apart(self):
        units, cfg = tcc.box(hf="/mnt/ssd/huggingface")
        items = plan(units, cfg)
        self.assertEqual(groups(items), {"HF_CACHE": ("site", "site")})
        line = conformity.plan_lines(items, "v1.36.0")[1]
        self.assertTrue(line.startswith("  site setting HF_CACHE /mnt/ssd/huggingface"), line)

    def test_this_command_line_wins_over_the_units(self):
        units, cfg = tcc.box(host="0.0.0.0", key_at=OLD_BOX)
        self.assertEqual(groups(plan(units, cfg, ENGINE_BIND="0.0.0.0")), {"ENGINE_BIND": ("yours", "env")})
        self.assertEqual(plan(units, cfg, ENGINE_BIND="127.0.0.1"), [], "the recommended value, set: nothing to list")

    def test_the_proxy_port_is_recommended_beside_the_port_this_run_serves(self):
        units, cfg = tcc.box(port="30010", proxy_port="30011")
        self.assertEqual(groups(plan(units, cfg)), {"PORT": ("site", "site")})
        self.assertEqual(groups(plan(units, cfg, PORT="30020")), {"PORT": ("site", "env"), "PROXY_PORT": ("site", "site")})

    def test_a_target_is_named_as_an_operator_says_it(self):
        units, cfg = tcc.box(flash_target="nvidia/Qwen3.8-Flash-Next-NVFP4")
        line = conformity.plan_lines(plan(units, cfg), "v1.36.0")[1]
        self.assertIn("stays yours  flash target flash-nvda (recommended: flash), no release installed it by default", line)

    def test_the_groups_come_in_order(self):
        units, cfg = tcc.box(host="0.0.0.0", key_at=OLD_BOX, hf="/mnt/ssd/huggingface", tier="throughput")
        labels = [ln[2:15].strip() for ln in conformity.plan_lines(plan(units, cfg), "v1.36.0")[1:]]
        self.assertEqual(labels, ["moves now", "stays yours", "site setting"])


def walled_run(units, cfg, home, args=()):
    """install.sh up to its first step on a box whose units are `units`, run with `home` as its
    home, whose config folder is `cfg` (copied there, its files' dates kept)."""
    shutil.copytree(cfg, pathlib.Path(home) / ".config/qwen38")
    script = wall.walled(units=units)
    env, record = wall.fenced_env(home=str(home))
    r = subprocess.run([script, *args], capture_output=True, text=True, env=env, cwd=wall.cwd(), timeout=120)
    assert not wall.reached(record), wall.reached(record)
    return r.stdout + r.stderr


class TheRunPrintsIt(unittest.TestCase):
    def setUp(self):
        self.home_was = tcc.HOME

    def tearDown(self):
        tcc.HOME = self.home_was

    def box_for(self, home, **kw):
        tcc.HOME = str(home)
        return tcc.box(**kw)

    def test_the_main_run_prints_the_list_before_step_1(self):
        home = wall._tmp("plan-home-")
        units, cfg = self.box_for(home, host="0.0.0.0", key_at=OLD_BOX)
        out = walled_run(units, cfg, home)
        self.assertIn(wall.WALL, out, "the run reached step 1")
        release = conformity.clone_release(REPO)
        at = out.index(f"This box against what {release} recommends:")
        self.assertLess(at, out.index(wall.WALL))
        self.assertRegex(out[at:], r"moves now +ENGINE_BIND 0\.0\.0\.0, to 127\.0\.0\.1: the default since v1\.17\.0")

    def test_a_box_as_recommended_gets_one_line(self):
        home = wall._tmp("plan-home-")
        units, cfg = self.box_for(home, key_at=OLD_BOX)
        out = walled_run(units, cfg, home)
        self.assertIn("recommends: the same, nothing of yours differs.", out)

    def test_the_run_for_the_other_text_lane_prints_none(self):
        home = wall._tmp("plan-home-")
        units, cfg = self.box_for(home, host="0.0.0.0", key_at=OLD_BOX)
        shutil.copytree(cfg, pathlib.Path(home) / ".config/qwen38")
        env, record = wall.fenced_env(home=str(home), QWEN38_SECONDARY="27b")
        r = subprocess.run([wall.walled(units=units)], capture_output=True, text=True, env=env, cwd=wall.cwd(),
                           timeout=120)
        out = r.stdout + r.stderr
        self.assertIn(wall.WALL, out, out[-600:])
        self.assertNotIn("This box against what", out)

    def run_plan(self, *args, files=()):
        home = wall._tmp("plan-home-")
        units, cfg = self.box_for(home, host="0.0.0.0", key_at=OLD_BOX)
        for name in files:
            (pathlib.Path(cfg) / name).write_text("left out\n")
        shutil.copytree(cfg, pathlib.Path(home) / ".config/qwen38")
        env, record = wall.fenced_env(home=str(home))
        r = subprocess.run([wall.walled(units=units), "--plan", *args], capture_output=True, text=True,
                           env=env, cwd=wall.cwd(), timeout=120)
        self.assertFalse(wall.reached(record), wall.reached(record))
        return r.returncode, r.stdout + r.stderr, pathlib.Path(home) / ".config/qwen38"

    def test_plan_lists_and_stops_before_anything_changes(self):
        rc, out, cfg = self.run_plan()
        self.assertEqual(rc, 0, out[-600:])
        self.assertNotIn(wall.WALL, out, "it stopped before step 1")
        self.assertRegex(out, r"moves now +ENGINE_BIND 0\.0\.0\.0, to 127\.0\.0\.1")
        self.assertIn("--plan: nothing was changed. The same command without --plan applies it.", out)
        self.assertFalse((cfg / "installed.json").exists(), "a plan records nothing")

    def test_plan_leaves_the_markers_as_they_are(self):
        before = ["cockpit.off", "video.off"]
        rc, out, cfg = self.run_plan("--no-image", "--with-cockpit", "--with-video", files=before)
        self.assertEqual(rc, 0, out[-600:])
        self.assertEqual(sorted(p.name for p in cfg.glob("*.off")), before)

    def test_plan_has_nothing_to_say_of_a_no_service_run(self):
        rc, out, _ = self.run_plan("--no-service")
        self.assertNotEqual(rc, 0)
        self.assertIn("--plan lists what a run does to the box's services; --no-service writes none.", out)

    def test_the_help_names_it(self):
        help_text = TEXT[TEXT.index("Usage: ./install.sh"):TEXT.index("HLP\n", TEXT.index("Usage: ./install.sh"))]
        self.assertIn("[--plan]", help_text)
        self.assertIn("  --plan                list this box against what this release recommends", help_text)

    def test_the_block_is_the_main_runs_and_never_fatal(self):
        block = TEXT[TEXT.index("# This box against what this release recommends"):TEXT.index('step "1/10 Preflight checks"')]
        self.assertIn('if [ -z "$SECONDARY" ] && [ "$NO_SERVICE" -eq 0 ]; then', block)
        self.assertRegex(block, r'--lane "\$LANE" \$\{CHOSE\[@\]\+"\$\{CHOSE\[@\]\}"\} \\\n    \|\| echo "NOTE: ')
        self.assertIn("--units /etc/systemd/system/", block, "the units the run reads, which the walled copies move")

    def test_record_install_and_the_list_take_the_same_choices(self):
        rec = TEXT[TEXT.index("record_install() {"):]
        rec = rec[:rec.index("\n}\n")]
        self.assertIn("collect_choices", rec)
        self.assertNotIn("--chose", rec.replace('${CHOSE[@]+"${CHOSE[@]}"}', ""),
                         "one list of explicit settings, not two that can drift apart")


if __name__ == "__main__":
    unittest.main(verbosity=2)
