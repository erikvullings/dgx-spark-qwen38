#!/usr/bin/env python3
"""The Flash FP4-Local lane reads its n-gram table from the disk of the home directory.

The table lives in seven of the checkpoint's files and is read at every token: from a USB SSD the
lane decoded 15.6 tokens a second for eight requests, from the internal NVMe 77.7 (reference box,
2026-10-19). ple-table.py copies those files next to the home directory when the cache is on
another disk, checks every copy against its blob's sha256, and gives the unit the mounts that put
the copies where the engine reads. These run it on a fake cache of a few kilobytes."""
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "ple-table.py"
NAME = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
HUB = "models--nvidia--Qwen3.8-Flash-Next-FP4-Local"


def load():
    spec = importlib.util.spec_from_file_location("ple_table", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCache:
    """A cache holding a checkpoint whose table is in two of its three files, as blobs named by sha256."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.blobs = root / "hub" / HUB / "blobs"
        self.snap = root / "hub" / HUB / "snapshots" / REV
        (self.snap / "tensors").mkdir(parents=True)
        self.blobs.mkdir(parents=True)
        self.files = {}
        weight_map = {}
        for i, (tensors, body) in enumerate([
                (["model.layers.0.mlp.experts.0.weight"], b"experts" * 100),
                (["model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight",
                  "model.language_model.layers.0.self_attn.q_proj.weight"], b"table-and-more" * 100),
                (["model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight"], b"table" * 100)]):
            name = f"tensors/model-0000{i + 1}-of-00003.safetensors"
            sha = hashlib.sha256(body).hexdigest()
            (self.blobs / sha).write_bytes(body)
            os.symlink(os.path.relpath(self.blobs / sha, (self.snap / name).parent), self.snap / name)
            self.files[name] = sha
            for t in tensors:
                weight_map[t] = name
        (self.snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        self.table = [self.files["tensors/model-00002-of-00003.safetensors"],
                      self.files["tensors/model-00003-of-00003.safetensors"]]


class TableFiles(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="ple-table-"))
        self.cache = FakeCache(self.d / "hf")
        self.home = self.d / "home"
        self.home.mkdir()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_it(self, action, *extra):
        return subprocess.run([sys.executable, "-I", str(SCRIPT), str(self.cache.root), NAME, REV,
                               "--home", str(self.home), *extra, action],
                              capture_output=True, text=True, timeout=60)

    def test_the_files_of_the_table_are_the_ones_its_tensors_are_in(self):
        mod = load()
        got = [b.name for _, b in mod.table_files(self.cache.snap)]
        self.assertEqual(got, self.cache.table)

    def test_a_cache_on_the_home_directorys_disk_is_read_in_place(self):
        r = self.run_it("plan")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("in place", r.stdout)
        self.assertEqual(self.run_it("mounts").stdout, "")

    def test_a_cache_on_another_disk_is_read_from_copies_in_the_home_directory(self):
        mod = load()
        real = os.stat

        def fake_stat(p, *a, **k):
            st = real(p, *a, **k)
            if pathlib.Path(p) == self.cache.blobs:
                return types.SimpleNamespace(st_dev=st.st_dev + 1)
            return st
        args = types.SimpleNamespace(dir="", home=str(self.home))
        with mock.patch.object(mod.os, "stat", fake_stat):
            self.assertEqual(mod.where(args, self.cache.blobs), self.home / "flash-local-ple")
        self.assertIsNone(mod.where(args, self.cache.blobs))
        self.assertIsNone(mod.where(types.SimpleNamespace(dir="none", home=str(self.home)), self.cache.blobs))

    def test_copies_are_checked_then_mounted_over_the_caches_blobs(self):
        folder = self.d / "copies"
        r = self.run_it("copy", "--dir", str(folder))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sorted(p.name for p in folder.iterdir() if not p.name.startswith(".")), sorted(self.cache.table))
        mounts = self.run_it("mounts", "--dir", str(folder)).stdout.split()
        want = [x for sha in self.cache.table for x in
                ("-v", f"{folder / sha}:/root/.cache/huggingface/hub/{HUB}/blobs/{sha}:ro")]
        self.assertEqual(mounts, want)

    def test_a_damaged_cache_file_is_never_copied(self):
        folder = self.d / "copies"
        (self.cache.blobs / self.cache.table[1]).write_bytes(b"bitrot" * 100)
        r = self.run_it("copy", "--dir", str(folder))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not match its sha256", r.stderr)
        self.assertFalse((folder / self.cache.table[1]).exists())
        self.assertFalse((folder / (self.cache.table[1] + ".part")).exists())

    def test_a_copy_that_changed_since_is_made_again(self):
        folder = self.d / "copies"
        self.assertEqual(self.run_it("copy", "--dir", str(folder)).returncode, 0)
        good = (folder / self.cache.table[0]).read_bytes()
        (folder / self.cache.table[0]).write_bytes(b"X" * len(good))
        self.assertEqual(self.run_it("copy", "--dir", str(folder)).returncode, 0)
        self.assertEqual((folder / self.cache.table[0]).read_bytes(), good)

    def new_revision_of_the_table(self):
        """The checkpoint's second table file changed, as a later revision would change it."""
        body = b"table, revised" * 100
        sha = hashlib.sha256(body).hexdigest()
        (self.cache.blobs / sha).write_bytes(body)
        link = self.cache.snap / "tensors/model-00003-of-00003.safetensors"
        link.unlink()
        os.symlink(os.path.relpath(self.cache.blobs / sha, link.parent), link)
        return [self.cache.table[0], sha]

    def listed(self, folder):
        return sorted(p.name for p in folder.iterdir() if not p.name.startswith("."))

    def test_an_earlier_revisions_copies_go_and_nothing_else(self):
        folder = self.d / "copies"
        self.assertEqual(self.run_it("copy", "--dir", str(folder)).returncode, 0)
        (folder / "notes.txt").write_text("mine")
        gone = self.cache.table[1]
        (folder / (gone + ".part")).write_bytes(b"cut short")
        table = self.new_revision_of_the_table()
        r = self.run_it("copy", "--dir", str(folder))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.listed(folder), sorted(table + ["notes.txt"]))

    def test_a_file_it_did_not_make_stays_whatever_its_name(self):
        # its copies are named by their sha256, as the cache's blobs are: a folder that holds other
        # such files (another model's blobs) lost them all at the first copy (found in review,
        # 2026-10-20)
        folder = self.d / "copies"
        folder.mkdir()
        theirs = ["b" * 64, "c" * 64 + ".part"]
        for name in theirs:
            (folder / name).write_bytes(b"not ours")
        self.assertEqual(self.run_it("copy", "--dir", str(folder)).returncode, 0)
        self.new_revision_of_the_table()
        self.assertEqual(self.run_it("copy", "--dir", str(folder)).returncode, 0)
        for name in theirs:
            self.assertEqual((folder / name).read_bytes(), b"not ours")

    def test_a_folder_inside_the_cache_is_refused_and_the_cache_kept(self):
        # pointed at the checkpoint's own blobs, the copy found the table's files already there and
        # removed every other blob of the checkpoint as an earlier revision's copies
        before = sorted(p.name for p in self.cache.blobs.iterdir())
        for folder in (self.cache.blobs, self.cache.root / "ple"):
            r = self.run_it("copy", "--dir", str(folder))
            self.assertNotEqual(r.returncode, 0, folder)
            self.assertIn("inside the Hugging Face cache", r.stdout + r.stderr)
        self.assertEqual(sorted(p.name for p in self.cache.blobs.iterdir()), before)
        self.assertFalse((self.cache.root / "ple").exists())

    def test_a_folder_the_unit_cannot_name_is_refused(self):
        # the unit names the copies inside bash -c '...', put in by sed: a space split the mount, a
        # quote ended the command, a & or a | broke the substitution, and the unit restarted in a loop
        # and a : split docker's -v into more fields than it takes (found in review, 2026-10-20):
        # install.sh's own rule for the paths a unit carries, the same
        for bad in ("copies here", "it's", "a&b", "a|b", "ple:2026", "a%b"):
            with self.subTest(folder=bad):
                for action in ("copy", "check"):
                    r = self.run_it(action, "--dir", str(self.d / bad))
                    self.assertNotEqual(r.returncode, 0)
                    self.assertIn("letters, digits and . _ / + @ - only", r.stdout + r.stderr)
                self.assertFalse((self.d / bad).exists())

    def test_check_needs_no_checkpoint_and_passes_a_plain_folder(self):
        shutil.rmtree(self.cache.snap)
        for good in ("none", "", str(self.d / "ple-copies_2.0+x@y")):
            with self.subTest(folder=good):
                r = self.run_it("check", "--dir", good)
                self.assertEqual(r.returncode, 0, r.stderr)

    def room(self, folder, free):
        mod = load()
        args = ["ple-table.py", str(self.cache.root), NAME, REV, "--home", str(self.home), "--dir", str(folder),
                "--need-gb", "33", "room"]
        with mock.patch.object(sys, "argv", args), \
             mock.patch.object(mod.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=free)):
            mod.main()

    def test_room_is_said_before_any_download(self):
        # the copy refused only after the checkpoint's 103 GB were fetched (found in review, 2026-10-20)
        folder = self.d / "copies"
        # with the checkpoint there, what is missing of the copies: a few KB here
        self.room(folder, free=2 * 2 ** 30)
        with self.assertRaises(SystemExit):
            self.room(folder, free=2 ** 29)
        # before it, the table's size
        shutil.rmtree(self.cache.snap)
        with self.assertRaises(SystemExit) as e:
            self.room(folder, free=20 * 10 ** 9)
        self.assertIn("need about 34 GB", str(e.exception))
        self.room(folder, free=40 * 10 ** 9)
        # read where the cache is, nothing to make room for
        self.room("none", free=0)

    def test_mounts_refuse_while_a_copy_is_missing(self):
        r = self.run_it("mounts", "--dir", str(self.d / "empty"))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")

    def test_no_snapshot_is_said_with_status_2(self):
        shutil.rmtree(self.cache.snap)
        r = self.run_it("plan")
        self.assertEqual(r.returncode, 2)
        self.assertIn("fetch the checkpoint first", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
