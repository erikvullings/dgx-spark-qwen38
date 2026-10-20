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

    def test_its_boot_leaves_the_box_its_memory(self):
        # vLLM loads the weights with PyTorch's allocator splitting no block above 20 MiB: the blocks b12x
        # freed while it laid out the experts' NVFP4 scales went unused, and the allocator's cache grew
        # until the GPU's driver refused it memory, which on this box is the machine's (the driver refused
        # 44 to 149 allocations at each of twelve boots, MemAvailable down to 1.05 GiB, 2026-10-20).
        # PyTorch collects its cached blocks only under a ceiling below the whole GPU, past the share of
        # it the threshold says: the ceiling above the share vLLM is given, the collection under it and
        # above what the engine holds in service (0.74 of the GPU at the card's share of 0.8)
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        cmd = exec_command(self.box.out.read_text())
        m = re.search(r"-e PYTORCH_CUDA_ALLOC_CONF=(\S+)", cmd)
        self.assertIsNotNone(m, "the engine's allocator is left as PyTorch sets it")
        conf = dict(kv.split(":", 1) for kv in m.group(1).split(","))
        ceiling, threshold = float(conf["per_process_memory_fraction"]), float(conf["garbage_collection_threshold"])
        share = float(re.search(r"--gpu-memory-utilization (\S+)", cmd).group(1))
        self.assertTrue(share < ceiling < 1.0, (share, ceiling))
        self.assertTrue(share - 0.05 <= ceiling * threshold < share, (ceiling, threshold, share))

    def test_copies_of_the_table_are_mounted_over_the_caches_files(self):
        folder = self.box.t / "copies"
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR=str(folder))
        self.assertEqual(rc, 0, out)
        self.assertEqual(sorted(p.name for p in folder.iterdir() if not p.name.startswith(".")), sorted(self.box.table))
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

    def test_it_writes_its_own_files_and_no_other_lanes(self):
        # the start scripts every lane runs are ./install.sh's: a Load put them in from the checkout,
        # moved past its install or not (found in review, 2026-10-20). The filter is the lane's own,
        # and an older one stays until ./install.sh as the scripts do.
        cfg = self.box.home / ".config" / "qwen38"
        for name in ("engine-preflight.sh", "engine-secrets.sh", "vllm-auth/vllm_engine_auth.py", "seccomp-io-uring.json"):
            (cfg / name).parent.mkdir(parents=True, exist_ok=True)
            (cfg / name).write_text("installed earlier\n")
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        for name in ("engine-preflight.sh", "engine-secrets.sh", "vllm-auth/vllm_engine_auth.py", "seccomp-io-uring.json"):
            self.assertEqual((cfg / name).read_text(), "installed earlier\n", name)
        for cache in ("vllm", "flashinfer", "b12x", "triton"):
            self.assertTrue((cfg / "flash-local-cache" / cache).is_dir(), cache)

    def test_its_room_is_checked_before_the_switch_and_the_install_fetch(self):
        r = subprocess.run(["bash", str(REPO / "install-flash-local.sh"), "--room"], capture_output=True, text=True,
                           timeout=120, env={"PATH": "/usr/bin:/bin", "HOME": str(self.box.home), "HF_CACHE": str(self.box.hf),
                                             "LANES_UNIT_DIR": str(self.box.units), "FLASH_LOCAL_PLE_DIR": str(self.box.t / "c")})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("room for the table's copies under", r.stdout)
        installer = (REPO / "install-flash-local.sh").read_text()
        self.assertLess(installer.rindex("\nlane_table_room\n"), installer.index('vlane_fetch "$MODEL" "$MODEL_REV"'))
        switch = (REPO / "switch-model.sh").read_text()
        block = switch[switch.index('if [ "$CHOICE" = "flash-local" ]; then'):]
        self.assertLess(block.index("install-flash-local.sh\" --room"), block.index("/opt/qwen38/ckpt-fetch.py"))

    def test_a_folder_refused_is_not_kept(self):
        # kept before it was checked, a folder ple-table.py refuses stayed the choice of every Load
        # after, each failing the same way (found in review, 2026-10-20)
        for bad in (str(self.box.hf / "ple"), str(self.box.t / "a:b")):
            with self.subTest(folder=bad):
                rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR=bad)
                self.assertNotEqual(rc, 0, out)
                self.assertFalse((self.box.home / ".config" / "qwen38" / "flash-local-ple-dir").exists())

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
