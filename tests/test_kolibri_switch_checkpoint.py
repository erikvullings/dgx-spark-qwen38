#!/usr/bin/env python3
"""The Kolibri lane's fetch (kolibri-fetch.py): a snapshot missing a shard is fetched, not taken as cached.

The switch first asks huggingface_hub for the snapshot offline. huggingface_hub 1.30 (the
lane image's) sees a missing file there only when it cached the repo's tree listing, which
a download made with another tool or an older hub never wrote: the NVFP4 checkpoint on the
reference box has no tree cache. A switch would then enable a unit whose vLLM, offline,
dies at every start. The script runs here, whole, against a fake hub. It is the one copy of the
fetch: install-kolibri.sh had its own, without that check (found in review, 2026-10-14).
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
FETCH = REPO / "kolibri-fetch.py"

FAKE_HUB = '''
import json, os, pathlib


class constants:
    HF_HUB_CACHE = os.path.join(os.environ["HF_HOME"], "hub")


def snapshot_download(repo, revision=None, local_files_only=False, **kw):
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(f"{repo} {revision} local={local_files_only}\\n")
    snap = pathlib.Path(os.environ["SNAP"])
    if not local_files_only:      # the online path completes the snapshot
        index = snap / "model.safetensors.index.json"
        names = json.loads(index.read_text())["weight_map"].values() if index.exists() else ["model.safetensors"]
        for name in names:
            (snap / name).write_text("w")
    return str(snap)
'''


class ACheckpointIsCachedOnlyWhenWhole(unittest.TestCase):
    def run_switch(self, files, index=None):
        d = pathlib.Path(tempfile.mkdtemp(prefix="kol-ckpt-"))
        (d / "huggingface_hub").mkdir()
        (d / "huggingface_hub" / "__init__.py").write_text(FAKE_HUB)
        snap = d / "hub" / "models--Quantinine--Kolibri-1-NVFP4A16-FP8" / "snapshots" / ("4" * 40)
        snap.mkdir(parents=True)
        if index is not None:
            (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
        for name in files:
            (snap / name).write_text("w")
        log = d / "calls.log"
        env = dict(os.environ, PYTHONPATH=str(d), HF_HOME=str(d), FAKE_LOG=str(log), SNAP=str(snap),
                   KOL_MODEL="Quantinine/Kolibri-1-NVFP4A16-FP8", KOL_REV="4" * 40,
                   KOL_NEED_GB="46", KOL_FREE_GB="1000")
        r = subprocess.run([sys.executable, str(FETCH)], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return log.read_text().splitlines(), r.stdout

    def test_a_snapshot_missing_a_shard_is_fetched(self):
        index = {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"}
        calls, out = self.run_switch(["model-00001-of-00002.safetensors"], index)
        self.assertEqual([c.split()[-1] for c in calls], ["local=True", "local=False"], calls)
        self.assertIn("not complete in the cache, fetching the rest", out)

    def test_a_whole_snapshot_needs_no_network(self):
        index = {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"}
        calls, out = self.run_switch(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"], index)
        self.assertEqual([c.split()[-1] for c in calls], ["local=True"], calls)
        self.assertNotIn("fetching", out)

    def test_a_single_file_checkpoint_is_read_the_same_way(self):
        calls, _ = self.run_switch([])
        self.assertEqual([c.split()[-1] for c in calls], ["local=True", "local=False"])
        calls, _ = self.run_switch(["model.safetensors"])
        self.assertEqual([c.split()[-1] for c in calls], ["local=True"])



class OneCopyOfTheFetch(unittest.TestCase):
    def test_both_callers_run_it_and_neither_has_its_own(self):
        switch = (REPO / "switch-model.sh").read_text()
        at = switch.index('if [ "$CHOICE" = "kolibri" ] || [ "$CHOICE" = "kolibri-nvfp4" ]; then')
        kolibri_block = switch[at:switch.index("\nfi\n", at)]
        for name, text in (("switch-model.sh, its Kolibri block", kolibri_block),
                           ("install-kolibri.sh", (REPO / "install-kolibri.sh").read_text())):
            with self.subTest(script=name):
                self.assertIn("/opt/qwen38/kolibri-fetch.py", text)
                self.assertNotIn("snapshot_download", text, "a fetch of its own again")


if __name__ == "__main__":
    unittest.main()
