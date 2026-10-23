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


class NothingUnmeasured(unittest.TestCase):
    def test_no_64_gb_value_ships_before_it_is_measured(self):
        # the values the 64 GB lanes started from carry this mark until a run on a 64 GB budget
        # replaced them: none may reach a release
        marked = subprocess.run(["git", "-C", str(REPO), "grep", "-l", "PROVISIONAL 64 GB VALUE"],
                                capture_output=True, text=True).stdout.split()
        self.assertEqual([m for m in marked if m != "tests/test_spark_64.py"], [])


if __name__ == "__main__":
    unittest.main()
