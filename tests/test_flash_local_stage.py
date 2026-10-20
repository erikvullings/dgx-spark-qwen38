#!/usr/bin/env python3
"""install-flash-local.sh --stage: the unit a switch to the Flash FP4-Local lane puts in place.

switch-model.sh flash-local fetches the checkpoint, then has the lane's installer write its unit
to a fixed staging path with no sudo (the one renderer, so a switch puts in the unit an install
would), after its chat template and the copies of its n-gram table. These run the installer as
written, in a throwaway home, on a fake cache whose table sits in two of its three files."""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_lean_level import STOCK  # noqa: E402  (the template patch-template.py knows how to patch)

PINS = {k: v.strip('"') for k, v in (line.split("=", 1) for line in (REPO / "install-flash-local.sh").read_text().splitlines()
                                       if line.startswith("FLASH_LOCAL_") and "=" in line)}
HUB = "models--" + PINS["FLASH_LOCAL_REPO"].replace("/", "--")


class Box:
    def __init__(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="flash-local-stage-"))
        self.home = self.t / "home"
        (self.home / ".config" / "qwen38").mkdir(parents=True)
        (self.home / ".config" / "qwen38" / "api-key").write_text("test-key\n")
        self.units = self.t / "units"
        self.units.mkdir()
        self.hf = self.t / "hf"
        blobs = self.hf / "hub" / HUB / "blobs"
        snap = self.hf / "hub" / HUB / "snapshots" / PINS["FLASH_LOCAL_REV"]
        (snap / "tensors").mkdir(parents=True)
        blobs.mkdir(parents=True)
        (snap / "chat_template.jinja").write_text(STOCK)
        weight_map, self.table = {}, []
        for i, (tensors, body) in enumerate([
                (["model.layers.0.mlp.experts.0.weight"], b"experts" * 64),
                (["model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"], b"table-0" * 64),
                (["model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight"], b"table-1" * 64)]):
            name = f"tensors/model-0000{i + 1}-of-00003.safetensors"
            sha = hashlib.sha256(body).hexdigest()
            (blobs / sha).write_bytes(body)
            os.symlink(os.path.relpath(blobs / sha, (snap / name).parent), snap / name)
            weight_map.update({t: name for t in tensors})
            if i:
                self.table.append(sha)
        (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        self.out = self.t / "stage.service"

    def stage(self, cwd=None, **env):
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "HF_CACHE": str(self.hf),
             "LANES_UNIT_DIR": str(self.units), **env}
        r = subprocess.run(["bash", str(REPO / "install-flash-local.sh"), "--stage", str(self.out)],
                           capture_output=True, text=True, env=e, timeout=120, cwd=cwd)
        return r.returncode, r.stdout + r.stderr

    def mounts(self):
        return re.findall(r"-v (\S+):/root/\.cache/huggingface/hub/\S+:ro", self.out.read_text())


def exec_command(unit: str) -> str:
    """The ExecStart command as bash runs it: systemd joins the lines a backslash ends."""
    line = unit[unit.index("ExecStart=/bin/bash -c '") + len("ExecStart=/bin/bash -c '"):]
    line = line[:line.index("'\nExecStop=")]
    return line.replace("\\\n", " ")


class TheStagedUnit(unittest.TestCase):
    def setUp(self):
        self.box = Box()

    def tearDown(self):
        shutil.rmtree(self.box.t, ignore_errors=True)

    def test_read_in_place_it_names_no_copy_and_is_whole(self):
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        self.assertNotRegex(unit, r"__[A-Z][A-Z0-9_]*__", "a placeholder left")
        self.assertNotIn("/blobs/", unit)
        self.assertIn(f"vllm serve {PINS['FLASH_LOCAL_REPO']} --revision {PINS['FLASH_LOCAL_REV']}", unit)
        self.assertIn(f"  {PINS['FLASH_LOCAL_IMAGE']} \\\n", unit)
        self.assertIn(f"--security-opt seccomp={self.box.home}/.config/qwen38/seccomp-io-uring.json", unit)
        self.assertNotIn("unconfined", unit)
        self.assertIn(f"-v {self.box.hf}:/root/.cache/huggingface", unit)
        self.assertIn("--served-model-name qwen3.8-flash-next-fp4", unit)
        self.assertIn("--host 127.0.0.1 --port 30000", unit)
        r = subprocess.run(["bash", "-n", "-c", exec_command(unit)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # what the unit reads at its start is there: the filter, the template with xhigh as its default
        cfg = self.box.home / ".config" / "qwen38"
        self.assertEqual((cfg / "seccomp-io-uring.json").read_bytes(), (REPO / "seccomp-io-uring.json").read_bytes())
        self.assertIn("reasoning_effort|default('xhigh')", (cfg / "chat-template-flash-local.jinja").read_text())

    def test_copies_of_the_table_are_mounted_over_the_caches_files(self):
        folder = self.box.t / "copies"
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR=str(folder))
        self.assertEqual(rc, 0, out)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted(self.box.table))
        unit = self.box.out.read_text()
        mounts = re.findall(r"-v (\S+):(/root/\.cache/huggingface/hub/\S+):ro", unit)
        self.assertEqual(mounts, [(str(folder / s), f"/root/.cache/huggingface/hub/{HUB}/blobs/{s}") for s in self.box.table])
        r = subprocess.run(["bash", "-n", "-c", exec_command(unit)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_folder_chosen_is_kept_for_a_load_run_with_none_of_the_users_environment(self):
        # a Load from the cockpit stages the unit with none of the user's environment: read without
        # the choice made at the install, it named no copy and the table was read where the cache is,
        # or copied it again to ~/flash-local-ple (found in review, 2026-10-20)
        folder = self.box.t / "copies"
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR=str(folder))
        self.assertEqual(rc, 0, out)
        rc, out = self.box.stage()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.box.mounts(), [str(folder / s) for s in self.box.table])
        self.assertFalse((self.box.home / "flash-local-ple").exists())
        # given empty, the default comes back: the cache here is on the home directory's disk
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.box.mounts(), [])
        rc, out = self.box.stage()
        self.assertEqual((rc, self.box.mounts()), (0, []), out)

    def test_a_relative_folder_is_mounted_by_its_whole_path(self):
        # docker run -v takes a host path only whole: "copies/<sha>" is read as a volume's name,
        # refused, and the unit restarted in a loop
        rc, out = self.box.stage(cwd=self.box.t, FLASH_LOCAL_PLE_DIR="copies")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.box.mounts(), [str(self.box.t / "copies" / s) for s in self.box.table])

    def test_a_cache_on_the_home_directorys_disk_is_read_in_place(self):
        # a new box: the cache under the home directory, on its disk
        rc, out = self.box.stage()
        self.assertEqual(rc, 0, out)
        self.assertNotIn("/blobs/", self.box.out.read_text())
        self.assertFalse((self.box.home / "flash-local-ple").exists())

    def test_without_the_checkpoint_it_refuses(self):
        shutil.rmtree(self.box.hf / "hub")
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertNotEqual(rc, 0, out)
        self.assertFalse(self.box.out.exists() and self.box.out.read_text().strip(), "a unit staged with no checkpoint")


if __name__ == "__main__":
    unittest.main(verbosity=2)
