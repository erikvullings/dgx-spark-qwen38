#!/usr/bin/env python3
"""A 64 GB DGX Spark gets the lanes that fit it, and every way in refuses the others before a byte.

The 64 GB DGX Spark ships from 2026-10-23 (boxmem.py says how it is told apart). install.sh sends
it the 27B in a unit sized for it and Bolt; the lanes' installers, the switch and the 27B's heavier
targets refuse the rest there, each before it fetches, pulls or writes anything. These run each
script's own lines with the class forced (QWEN38_MEMORY_CLASS), so no installer runs here."""
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SIDE = {"install-image.sh": "image", "install-video.sh": "video", "install-kolibri.sh": "kolibri",
        "install-nemotron-super.sh": "nemotron-super", "install-flash-local.sh": "flash-local",
        "install-ornith-122b.sh": "ornith-122b"}
SOURCE = '. "$HERE/lanes.sh"\n'


def bash(script, cls, **env):
    """script after lanes.sh and install.sh's die(), the class forced, in a throwaway HOME."""
    die = next(ln for ln in (REPO / "install.sh").read_text().splitlines() if ln.startswith("die()"))
    clean = {k: v for k, v in os.environ.items() if not k.startswith("QWEN38_")}
    with tempfile.TemporaryDirectory(prefix="spark64-home-") as home:
        r = subprocess.run(["bash", "-c", f'set -euo pipefail\n. "{REPO}/lanes.sh"\n{die}\n{script}'],
                           capture_output=True, text=True, timeout=30,
                           env={**clean, "QWEN38_MEMORY_CLASS": cls, "HOME": home, **env})
    return r.returncode, r.stdout, r.stderr


class TheSideLanes(unittest.TestCase):
    def guard(self, script):
        text = (REPO / script).read_text()
        self.assertEqual(text.count(SOURCE), 1, script)
        after = text.split(SOURCE, 1)[1].splitlines()
        # the comment, then the refusal: the first thing the script does once it has the library
        self.assertTrue(after[0].startswith("# A 64 GB DGX Spark cannot hold this lane"), after[0])
        self.assertTrue(after[2].startswith('[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits '), after[2])
        return after[2]

    def test_each_refuses_a_64_gb_box_and_lets_a_128_gb_one_through(self):
        for script, lane in SIDE.items():
            with self.subTest(script):
                line = self.guard(script)
                self.assertIn(f"lane_fits {lane} ||", line)
                rc, out, err = bash('ACTION=install\n' + line + '\necho "went on"', "64")
                self.assertEqual(rc, 1, err)
                self.assertNotIn("went on", out)
                self.assertIn("needs a 128 GB DGX Spark, and this box is a 64 GB one.", err)
                self.assertEqual(bash('ACTION=install\n' + line + '\necho "went on"', "128")[:2], (0, "went on\n"))
                # its --uninstall takes away what an older release put there, on a 64 GB box too
                self.assertEqual(bash('ACTION=uninstall\n' + line + '\necho "went on"', "64")[:2], (0, "went on\n"))

    def test_nothing_runs_before_the_refusal_but_the_script_finding_itself(self):
        # what comes before the library: comments, set, die(), the arguments and HERE, no command
        # that fetches, pulls, writes a unit or asks for root
        for script in SIDE:
            with self.subTest(script):
                head = (REPO / script).read_text().split(SOURCE, 1)[0]
                for ln in head.splitlines():
                    self.assertNotRegex(ln.strip(), r"^(docker|sudo|curl|wget|pip3?|systemctl|hf|huggingface-cli) ",
                                        f"{script} runs this before the refusal")


class Bolt(unittest.TestCase):
    def test_bolt_takes_the_share_of_its_class(self):
        text = (REPO / "install-nemotron.sh").read_text()
        pins = "\n".join(ln for ln in text.splitlines() if re.match(r"NEMOTRON_BOLT_(GMU|GMU_64|GB)=", ln))
        pick = next(ln for ln in text.splitlines() if ln.startswith('GMU="$NEMOTRON_BOLT_GMU"'))
        pick64 = next(ln for ln in text.splitlines() if "GMU=\"$NEMOTRON_BOLT_GMU_64\"" in ln)
        for cls, want in (("128", "0.70"), ("64", "0.60")):
            with self.subTest(cls):
                rc, out, err = bash(f'{pins}\n{pick}\n{pick64}\necho "$GMU"', cls)
                self.assertEqual((rc, out), (0, want + "\n"), err)

    def test_a_switch_on_a_64_gb_box_writes_its_share(self):
        text = (REPO / "switch-model.sh").read_text()
        self.assertIn('NEMOTRON_BOLT_GMU_64', (REPO / "install-nemotron.sh").read_text())
        self.assertIn("|GMU|GMU_64|GB)=", text, "the switch reads the 64 GB share with the others")
        line = next(ln for ln in text.splitlines() if 'eval "NEM_GMU=' in ln and "_GMU_64" in ln)
        rc, out, err = bash(f'NEM_PIN=NEMOTRON_BOLT; NEMOTRON_BOLT_GMU_64=0.60; NEM_GMU=0.70\n{line}\necho "$NEM_GMU"', "64")
        self.assertEqual((rc, out), (0, "0.60\n"), err)
        rc, out, err = bash(f'NEM_PIN=NEMOTRON_BOLT; NEMOTRON_BOLT_GMU_64=0.60; NEM_GMU=0.70\n{line}\necho "$NEM_GMU"', "128")
        self.assertEqual((rc, out), (0, "0.70\n"), err)


class TheSwitch(unittest.TestCase):
    def test_a_target_a_64_gb_box_cannot_hold_is_refused_before_anything(self):
        text = (REPO / "switch-model.sh").read_text()
        src = '. "$REPO_DIR/lanes.sh"\n'
        after = text.split(src, 1)[1].splitlines()
        self.assertTrue(after[1].startswith('[ "$(box_mem_class)" != 64 ] || target_fits "$CHOICE"'), after[1])
        for choice, goes in (("stock", True), ("uncensored", True), ("nemotron", True), ("fp8", False),
                             ("uncensored-fp8", False), ("flash", False), ("image-turbo", False),
                             ("video", False), ("kolibri-nvfp4", False), ("nemotron-super", False),
                             ("flash-local", False), ("ornith-122b", False)):
            with self.subTest(choice):
                rc, out, err = bash(f'CHOICE={choice}\n{after[1]}\necho "went on"', "64")
                self.assertEqual(rc == 0 and out == "went on\n", goes, err)
                if not goes:
                    self.assertIn(f"The {choice} target needs a 128 GB DGX Spark", err)
                self.assertEqual(bash(f'CHOICE={choice}\n{after[1]}\necho "went on"', "128")[:2], (0, "went on\n"))


class The27BUnit(unittest.TestCase):
    def test_a_64_gb_box_gets_the_27b_unit_sized_for_it(self):
        text = (REPO / "install.sh").read_text()
        i = text.index('  UNIT_TPL="$REPO_DIR/qwen38-sglang.service.template"\n')
        block = text[i:text.index("\n", text.index("qwen38-sglang-64.service.template", i)) + 1]
        for cls, ctx, want in (("64", "native", "qwen38-sglang-64.service.template"),
                               ("128", "native", "qwen38-sglang.service.template"),
                               ("128", "1m", "qwen38-sglang-1m.service.template")):
            with self.subTest((cls, ctx)):
                rc, out, err = bash(f'REPO_DIR=R; MEM_CLASS={cls}; CONTEXT_MODE={ctx}\n{block}echo "$UNIT_TPL"', cls)
                self.assertEqual((rc, out), (0, f"R/{want}\n"), err)

    def test_the_conformity_list_recommends_the_window_a_box_holds(self):
        for cls, want in (("64", "native"), ("128", "1m")):
            with self.subTest(cls):
                r = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                                    "import conformity; print(conformity.recommended()['CONTEXT_MODE'])", str(REPO)],
                                   capture_output=True, text=True, env={**os.environ, "QWEN38_MEMORY_CLASS": cls})
                self.assertEqual(r.stdout.strip(), want, r.stderr)


INSTALL = (REPO / "install.sh").read_text()


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


class TheRunSaysIt(unittest.TestCase):
    """What install.sh says of a 64 GB box, run from its own lines (found in review, 2026-10-23)."""
    def too_big_fn(self):
        return between(INSTALL, "too_big(){", "\n")

    def test_a_lane_the_box_cannot_hold_needs_128_gb_and_was_left_out_by_no_flag(self):
        lane_line = between(INSTALL, "    lane_line(){", "\n    }\n")
        script = (f"TOO_BIG=(video flash)\nCOCKPIT=1\n{self.too_big_fn()}{lane_line}"
                  'lane_line "Videos     " off video video "174 GB" /nonexistent\n'
                  'lane_line "Images     " off image image "46 GB" /nonexistent\n')
        rc, out, err = bash(script, "64")
        self.assertEqual(rc, 0, err)
        self.assertIn("Videos     : needs a 128 GB DGX Spark, and this box is a 64 GB one", out)
        self.assertNotIn("--no-video", out)
        self.assertIn("Images     : left out by --no-image on this run", out)

    def test_the_other_text_lane_too(self):
        block = between(INSTALL, '    case "$SECOND_STATE" in\n      installed) echo "  the ${SECOND_LANE/27b/27B} lane is up to date',
                        "\n    esac\n")
        rc, out, err = bash(f"TOO_BIG=(flash)\nSECOND_LANE=flash\nSECOND_STATE=off\nSECOND_CANNOT=\n{self.too_big_fn()}{block}", "64")
        self.assertEqual((rc, out.strip()), (0, "the flash lane needs a 128 GB DGX Spark, and this box is a 64 GB one"), err)

    def test_a_27b_unit_serving_a_model_of_its_own_still_updates(self):
        block = between(INSTALL, 'if [ "$MEM_CLASS" = 64 ]; then\n  # run.sh serves', "\nfi\n")
        for choice, goes in (("custom", True), ("stock", True), ("fp8", False)):
            with self.subTest(choice):
                rc, out, err = bash(f"MEM_CLASS=64\nNO_SERVICE=0\nLANE=27b\nCONTEXT_MODE=\nMODEL_CHOICE={choice}\n{block}"
                                    'echo "went on"', "64")
                self.assertEqual(rc == 0 and out == "went on\n", goes, err)
                self.assertNotIn("invalid choice", err)

    def test_no_service_is_refused_as_run_sh_serves_a_128_gb_box(self):
        block = between(INSTALL, 'if [ "$MEM_CLASS" = 64 ]; then\n  # run.sh serves', "\nfi\n")
        rc, out, err = bash(f"MEM_CLASS=64\nNO_SERVICE=1\nLANE=27b\nCONTEXT_MODE=\nMODEL_CHOICE=stock\n{block}"
                            'echo "went on"', "64")
        self.assertEqual(rc, 1, err)
        self.assertIn("--no-service prepares ./run.sh, which runs the 27B with the memory settings of a 128 GB DGX Spark", err)

    def test_a_forced_class_that_is_no_class_is_refused_by_its_name(self):
        for cls, ok in (("64", True), ("128", True), ("", True), ("64GB", False), ("96", False)):
            with self.subTest(cls):
                env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN38_")}
                r = subprocess.run(["bash", "-c", f'. "{REPO}/lanes.sh"\necho sourced'], capture_output=True,
                                   text=True, timeout=30, env={**env, "QWEN38_MEMORY_CLASS": cls})
                self.assertEqual((r.returncode, r.stdout), (0, "sourced\n") if ok else (1, ""), r.stderr)
                if not ok:
                    self.assertIn(f"QWEN38_MEMORY_CLASS={cls} is not a memory class: 64 or 128", r.stderr)

    def test_prepare_lane_refuses_what_a_64_gb_box_cannot_hold_before_a_pull(self):
        # the cockpit's Load runs it before its switch, which refused these only after the image
        # and the draft were pulled (found in review, 2026-10-23)
        env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN38_")}
        for args, said in ((("27b", "fp8"), "The fp8 target needs a 128 GB DGX Spark"),
                           (("flash",), "The flash lane needs a 128 GB DGX Spark")):
            with self.subTest(args), tempfile.TemporaryDirectory(prefix="prep64-") as home:
                r = subprocess.run(["bash", str(REPO / "prepare-lane.sh"), *args], capture_output=True, text=True,
                                   timeout=60, env={**env, "HOME": home, "PATH": "/usr/bin:/bin", "QWEN38_MEMORY_CLASS": "64"})
                self.assertEqual(r.returncode, 1, r.stderr[-400:])
                self.assertIn(said, r.stderr)

    def test_the_lanes_that_fit_come_from_boxmem(self):
        r = subprocess.run([sys.executable, "-I", str(REPO / "boxmem.py"), "fitting", "--class", "64"],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.split(), ["27b", "nemotron", "speech"])
        self.assertNotIn("the 27B and Nemotron 3.5 Bolt here", INSTALL, "a second list of them in install.sh")

class NothingUnmeasured(unittest.TestCase):
    def test_no_64_gb_value_ships_before_it_is_measured(self):
        # the values the 64 GB lanes started from carry this mark until a run on a 64 GB budget
        # replaced them: none may reach a release
        marked = subprocess.run(["git", "-C", str(REPO), "grep", "-l", "PROVISIONAL 64 GB VALUE"],
                                capture_output=True, text=True).stdout.split()
        self.assertEqual([m for m in marked if m != "tests/test_spark_64.py"], [])


if __name__ == "__main__":
    unittest.main()
