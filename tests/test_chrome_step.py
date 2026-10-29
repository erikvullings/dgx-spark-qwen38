#!/usr/bin/env python3
"""ci.yml's Chrome step installs a browser only where there is none the headless checks would take.

The checks (resilience, update-banner, monkey and touch) take CHROME, or the first of google-chrome,
chromium and chromium-browser. The step looked for google-chrome alone: ci-local.sh runs it on a
contributor's machine too, and one with only Chromium went to Google's repository and sudo apt
(review of PR #60, 2026-10-29). The step runs here as ci.yml writes it, with a PATH of stubs that
record their calls: sudo, curl, gpg and apt-get must not be called where a browser is."""
import pathlib
import subprocess
import tempfile
import unittest

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

REPO = pathlib.Path(__file__).resolve().parents[1]
STEP = "Google Chrome for arm64 (the arm64 runner has none)"


def step_script():
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    for job in wf["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == STEP:
                return step["run"]
    raise AssertionError(f"no step named {STEP!r} in ci.yml")


@unittest.skipUnless(HAVE_YAML, "the step is read from ci.yml with PyYAML, which this python lacks")
class TheChromeStep(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.tmp)]))
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.calls = self.tmp / "calls"
        for tool in ("sudo", "curl", "gpg", "apt-get", "tee", "awk"):
            self.stub(tool, f'echo {tool} >> "{self.calls}"\nexit 1\n')

    def stub(self, name, body, where=None):
        p = (where or self.bin) / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(0o755)
        return p

    def run_step(self, **env):
        return subprocess.run(["/bin/bash", "-e", "-c", step_script()], capture_output=True, text=True, timeout=60,
                              env={"PATH": str(self.bin), "HOME": str(self.tmp), "TMPDIR": str(self.tmp), **env})

    def called(self):
        return self.calls.read_text().split() if self.calls.exists() else []

    def test_a_machine_with_only_chromium_installs_nothing(self):
        self.stub("chromium", 'echo "Chromium 140.0"\n')
        r = self.run_step()
        self.assertEqual(self.called(), [], r.stdout + r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Chromium 140.0", r.stdout)

    def test_chrome_set_by_the_caller_installs_nothing(self):
        other = self.tmp / "elsewhere"
        other.mkdir()
        browser = self.stub("my-chrome", 'echo "Google Chrome 141.0"\n', where=other)
        r = self.run_step(CHROME=str(browser))
        self.assertEqual(self.called(), [], r.stdout + r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Google Chrome 141.0", r.stdout)

    def test_a_machine_with_no_browser_goes_to_install_one(self):
        r = self.run_step()
        self.assertIn("curl", self.called(), r.stdout + r.stderr)   # Google's key first, as on the runner
        self.assertNotEqual(r.returncode, 0)                         # the stubbed curl fails the install


if __name__ == "__main__":
    unittest.main(verbosity=2)
