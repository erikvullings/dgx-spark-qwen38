#!/usr/bin/env python3
"""The vLLM lanes that patch their chat template write it through one function, also on an install of
the units alone when the pinned checkpoint is in the cache.

The Flash FP4-Local and Ornith lanes' units mount their patched template, and their installers' --units-only
path put the unit without writing it: an update that took that path on a box that had served the lane (its
image pruned since) left a unit whose start docker refuses (code review, 2026-10-24). vllm-lane.sh's own
vlane_template_if_cached runs here on a throwaway cache, with patch-template.py itself."""
import importlib.util
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("patch_template", REPO / "patch-template.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)
MODEL, REV = "org/lane", "f" * 40


class TheTemplateOfAnInstallOfTheUnitsAlone(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="vlane-template-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.out = self.d / "chat-template-lane.jinja"
        self.body = "HEAD\n" + pt.SYSTEM_ANCHOR + "\nTAIL\n"

    def cache(self):
        snap = self.d / "hf" / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots" / REV
        snap.mkdir(parents=True)
        (snap / "chat_template.jinja").write_text(self.body)

    def run_it(self, *fixes):
        script = "\n".join([
            "set -euo pipefail", 'die(){ echo "die: $*"; exit 1; }', f'HERE="{REPO}"',
            f'. "{REPO}/vllm-lane.sh"', f'HF_CACHE="{self.d}/hf"', f"MODEL={MODEL}", f"MODEL_REV={REV}",
            f'vlane_template_if_cached "{self.out}" ' + " ".join(fixes), "echo DONE"])
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                              env=dict(os.environ, PATH="/usr/bin:/bin"))

    def test_written_with_the_lanes_fixes_when_the_checkpoint_is_cached(self):
        self.cache()
        r = self.run_it("--patches=system-reminder")
        self.assertIn("DONE", r.stdout, r.stdout + r.stderr)
        self.assertEqual(self.out.read_text(), self.body.replace(pt.SYSTEM_ANCHOR, pt.SYSTEM_PATCHED))

    def test_nothing_written_and_no_failure_without_the_checkpoint(self):
        r = self.run_it("--patches=system-reminder")
        self.assertIn("DONE", r.stdout, r.stdout + r.stderr)
        self.assertFalse(self.out.exists())

    def test_a_fix_that_is_neither_a_variable_nor_an_option_stops_it(self):
        self.cache()
        r = self.run_it("system-reminder")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("neither VAR=value nor --option", r.stdout)


if __name__ == "__main__":
    unittest.main()
