#!/usr/bin/env python3
"""A checkpoint with one file over the CDN limit must still download.

This repo disables the Hub's Xet transfer backend for a measured reason: during
the release campaign it stalled at 0-8 MB/s, sometimes at zero bytes forever,
while the classic CDN moved 89 MB/s on the same box in the same second.

That ban has one casualty. The classic CDN refuses any single file over
MAX_HTTP_DOWNLOAD_SIZE (50,000,000,000 bytes) with a ValueError that names
hf_xet, and the `flash-nvda` target ships its n-gram table as one 53.7 GB
safetensors. Seen 2026-09-18: that target could not be fetched at all with this
repo's settings. The resume loop retried the refusal four times and the run
died, over a traceback that did name hf_xet, under a failure message listing
four causes that were all the wrong one, with 74 GB of the 124 left in cache.

So the fetch (ckpt-fetch.py, which install.sh step 4 and switch-model.sh step 1 both run,
since v1.31 the one copy) turns Xet back on for exactly the repo the CDN just refused, and
turns it off again. These gates run the script against a stub Hub: they fail if the retry
is removed, if it fires on unrelated failures, or if it leaves Xet on for the rest of the run.
"""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOO_LARGE = ("The file is too large to be downloaded using the regular download method. "
             " Install `hf_xet` with `pip install hf_xet` for xet-powered downloads.")

# A stub Hub. It records every call with the state of the Xet switch at that
# moment, which is the fact these gates are about.
FAKE_HUB = '''
import json
import os


class _Constants:
    HF_HUB_DISABLE_XET = os.environ.get("HF_HUB_DISABLE_XET") == "1"
    MAX_HTTP_DOWNLOAD_SIZE = 50_000_000_000


constants = _Constants()


def _calls():
    log = os.environ["FAKE_LOG"]
    if not os.path.exists(log):
        return []
    return [json.loads(l) for l in open(log) if l.strip()]


def snapshot_download(repo, revision=None):
    seen = _calls()
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"repo": repo, "rev": revision,
                            "xet_disabled": constants.HF_HUB_DISABLE_XET}) + "\\n")
    mode = os.environ["FAKE_MODE"]
    mine = [c for c in seen if c["repo"] == repo]
    if mode == "cdn_limit" and repo == os.environ["FAKE_BIG_REPO"]:
        if constants.HF_HUB_DISABLE_XET:
            raise ValueError("The file is too large to be downloaded using the regular "
                             "download method.  Install `hf_xet` with `pip install hf_xet` "
                             "for xet-powered downloads.")
    elif mode == "cdn_limit_always" and repo == os.environ["FAKE_BIG_REPO"]:
        raise ValueError("The file is too large to be downloaded using the regular "
                         "download method.  Install `hf_xet` with `pip install hf_xet` "
                         "for xet-powered downloads.")
    elif mode == "other_error" and not mine:
        raise OSError("connection reset by peer")
    return os.environ["FAKE_SNAPSHOT"]
'''

# The block sleeps 10 s between resumed attempts. Tests measure what it does,
# not how long it waits.
NO_SLEEP = "import time\ntime.sleep = lambda *a, **k: None\n"


FETCH = REPO / "ckpt-fetch.py"


class XetRetry(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="xet-gate-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "huggingface_hub.py").write_text(FAKE_HUB)
        (self.tmp / "sitecustomize.py").write_text(NO_SLEEP)
        sha = "f" * 40
        self.snapshot = self.tmp / "hub" / "models--x--y" / "snapshots" / sha
        self.snapshot.mkdir(parents=True)
        self.log = self.tmp / "calls.jsonl"

    def fetch(self, repo, *, mode, big_repo):
        env = dict(os.environ,
                   PYTHONPATH=str(self.tmp),
                   HF_HOME=str(self.tmp),
                   HF_HUB_DISABLE_XET="1",           # what every caller sets
                   FAKE_MODE=mode,
                   FAKE_BIG_REPO=big_repo,
                   FAKE_LOG=str(self.log),
                   FAKE_SNAPSHOT=str(self.snapshot))
        r = subprocess.run([sys.executable, str(FETCH), repo, "rev1", "--refs-main", "fill"],
                           capture_output=True, text=True, timeout=120, env=env)
        calls = [json.loads(l) for l in self.log.read_text().splitlines() if l.strip()] \
            if self.log.exists() else []
        return r, calls

    def test_an_oversized_repo_is_retried_with_xet(self):
        """The CDN refusal is answered by one retry with Xet on."""
        r, calls = self.fetch("org/model", mode="cdn_limit", big_repo="org/model")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual([c["xet_disabled"] for c in calls], [True, False],
                         f"expected a CDN attempt then a Xet attempt, got {calls}")
        self.assertIn("retrying this repo with Xet", r.stdout)

    def test_xet_stays_off_for_the_next_repo(self):
        """The exception is per repo. A 27B install fetches two, one run each, and only one is huge."""
        self.fetch("org/model", mode="cdn_limit", big_repo="org/model")
        r, calls = self.fetch("org/draft", mode="cdn_limit", big_repo="org/model")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        draft_calls = [c for c in calls if c["repo"] == "org/draft"]
        self.assertTrue(draft_calls, f"the drafter was never downloaded: {calls}")
        self.assertTrue(all(c["xet_disabled"] for c in draft_calls),
                        f"the drafter inherited Xet from the repo before it: {draft_calls}")

    def test_xet_goes_back_off_after_the_retry(self):
        """Within one process: on again for the refused repo, off again right after."""
        sys.path.insert(0, str(self.tmp))
        self.addCleanup(sys.path.remove, str(self.tmp))
        os.environ.update(HF_HUB_DISABLE_XET="1", HF_HOME=str(self.tmp), FAKE_MODE="cdn_limit",
                          FAKE_BIG_REPO="org/model", FAKE_LOG=str(self.log), FAKE_SNAPSHOT=str(self.snapshot))
        for k in ("HF_HUB_DISABLE_XET", "HF_HOME", "FAKE_MODE", "FAKE_BIG_REPO", "FAKE_LOG", "FAKE_SNAPSHOT"):
            self.addCleanup(os.environ.pop, k, None)
        sys.modules.pop("huggingface_hub", None)
        self.addCleanup(sys.modules.pop, "huggingface_hub", None)
        spec = importlib.util.spec_from_file_location("ckpt_fetch_under_test", FETCH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.download("org/model", "rev1", {})
        self.assertTrue(mod.constants.HF_HUB_DISABLE_XET, "Xet left on after the retry")

    def test_an_unrelated_failure_does_not_turn_xet_on(self):
        """A reset connection is what the resume loop is for; it says nothing about transports."""
        r, calls = self.fetch("org/model", mode="other_error", big_repo="none")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(all(c["xet_disabled"] for c in calls),
                        f"an unrelated error flipped the transport: {calls}")
        self.assertIn("resuming", r.stdout)

    def test_a_repo_xet_cannot_fetch_either_still_fails(self):
        """The retry must not become a loop that never ends or a failure that reads as success."""
        r, calls = self.fetch("org/model", mode="cdn_limit_always", big_repo="org/model")
        self.assertNotEqual(r.returncode, 0, "an undownloadable repo reported success")
        self.assertNotIn(str(self.snapshot), r.stdout)
        self.assertLessEqual(len(calls), 12, f"the retry compounded into {len(calls)} attempts")

    def test_both_text_downloaders_run_the_one_script(self):
        """Two copies of one rule drift, and this pair did once: both run ckpt-fetch.py now."""
        for script in ("install.sh", "switch-model.sh"):
            text = (REPO / script).read_text()
            with self.subTest(script=script):
                self.assertIn("/opt/qwen38/ckpt-fetch.py", text)
                self.assertNotIn("def download(", text, "a download helper of its own again")


if __name__ == "__main__":
    unittest.main()
