#!/usr/bin/env python3
"""The installers tell which disk each need lands on by the device of its directory.

The model cache, Docker's root, a lane's runtime and the flash lane's PLE table can each be on
their own disk, and the checks sum what lands on one. On a box whose ~/.cache/huggingface is a
symlink to another disk (the reference box since 2026-10-13: an external SSD), `stat -c %d` read
the link's own device, the home directory's: install.sh took the cache and Docker for one disk
and checked their sum against the SSD's free space, the flash lane counted its 47.7 GiB PLE table
on the SSD, install-image.sh checked the runtime's room on the SSD, and install-video.sh mixed the
two disks' needs. `stat -L` reads the device the directory really is on.
"""
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
# every shell script of the repository: the vLLM lanes' room moved from their installers to
# vllm-lane.sh, which a fixed list missed (found in review, 2026-10-16)
SCRIPTS = sorted(p.name for p in REPO.glob("*.sh"))


class TheDiskChecksFollowSymlinks(unittest.TestCase):
    def test_no_device_comparison_reads_a_link_itself(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                bad = [f"{i}: {ln.strip()}" for i, ln in enumerate((REPO / name).read_text().splitlines(), 1)
                       if re.search(r"\bstat\s+-c\s+%d", ln)]
                self.assertEqual(bad, [], f"{name} compares the device of a symlink, not of its target")

    def test_the_scripts_that_compare_disks_follow_the_links(self):
        # and the comparison is there to be checked: none of these passes for want of one
        for name in ("install.sh", "install-image.sh", "install-video.sh", "vllm-lane.sh", "lanes.sh"):
            with self.subTest(script=name):
                self.assertRegex((REPO / name).read_text(), r"\bstat\s+-L\s+-c\s+%d", name)

    def test_a_link_to_another_disk_is_on_that_disk(self):
        # why the flag matters: /dev/shm is a tmpfs, another device than a temporary directory
        other = tempfile.mkdtemp(dir="/dev/shm", prefix="disk-check-")
        home = pathlib.Path(tempfile.mkdtemp(prefix="disk-check-home-"))
        link = home / "huggingface"
        os.symlink(other, link)
        dev = lambda *a: subprocess.run(["stat", *a, "-c", "%d", str(link)], capture_output=True, text=True).stdout.strip()
        self.assertEqual(dev(), str(os.stat(home).st_dev), "without -L: the home directory's device")
        self.assertEqual(dev("-L"), str(os.stat(other).st_dev), "with -L: the disk the cache is on")
        self.assertNotEqual(dev(), dev("-L"))
        os.rmdir(other)


if __name__ == "__main__":
    unittest.main()
