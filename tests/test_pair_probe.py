#!/usr/bin/env python3
"""pair-probe.py, the read-only probe a pair of Spark boxes runs before a dual-box lane (#45).

What it prints is pasted into a public discussion, so the invariant is what it never prints: a MAC
address, an IP address (only its kind), the home directory, the host's name, the box's serial
number, or anything of a file that holds a key, which it never opens."""
import builtins
import importlib.util
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "pair-probe.py"


def load():
    spec = importlib.util.spec_from_file_location("pair_probe", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = load()
MAC = re.compile(r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b", re.I)
IPV4 = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")


class Masking(unittest.TestCase):
    def test_an_address_is_only_its_kind(self):
        self.assertEqual([P.ip_kind(a) for a in ("192.168.100.2", "10.0.0.7/24", "172.20.1.1", "169.254.3.4",
                                                 "100.114.54.60", "8.8.8.8", "127.0.0.1", "not-an-ip")],
                         ["private", "private", "private", "link-local", "shared", "public", "loopback", "unknown"])

    def test_what_names_the_box_or_its_owner_goes(self):
        text = (f"inet 192.168.100.2/24 brd 192.168.100.255 link/ether a0:88:c2:12:34:56 {P.HOME}/x "
                "on gx10-eff9 driver 580.178.04.1 dgx-spark-qwen38")
        out = P.masked(text, host="gx10-eff9")
        self.assertNotRegex(out, MAC)
        self.assertNotRegex(out, IPV4)
        self.assertNotIn(P.HOME, out)
        self.assertNotIn("gx10-eff9", out)
        self.assertIn("<private>", out)
        self.assertIn("580.178.04.1", out, "a version is not an address")

    def test_a_host_named_like_a_word_of_the_project_eats_nothing(self):
        out = P.masked("dgx-spark-qwen38 on spark, DGX Spark", host="spark")
        self.assertEqual(out, "dgx-spark-qwen38 on <host>, DGX Spark")


class TheBox(unittest.TestCase):
    def test_the_version_is_the_last_update_and_the_serial_is_never_read_out(self):
        rel = pathlib.Path(tempfile.mkdtemp()) / "dgx-release"
        rel.write_text('DGX_NAME="DGX Spark"\nDGX_SWBUILD_VERSION="7.2.3"\nDGX_PLATFORM="GX10"\n'
                       'DGX_SERIAL="SN-1234567890"\n\nDGX_OTA_VERSION="7.3.1"\n\nDGX_OTA_VERSION="7.6.0"\n')
        facts = P.box(release=str(rel))
        self.assertEqual((facts["dgx_os"], facts["platform"]), ("7.6.0", "GX10"))
        self.assertNotIn("SN-1234567890", P.report({"box": facts}))
        self.assertNotIn("SN-1234567890", json.dumps(facts))


class ThePorts(unittest.TestCase):
    def test_the_connectx7_ports_and_their_rdma_side(self):
        root = pathlib.Path(tempfile.mkdtemp())
        drv = root / "drivers" / "mlx5_core"
        drv.mkdir(parents=True)
        for name, driver, state in (("enp1s0f0np0", drv, "up"), ("enP7s7", None, "up")):
            d = root / "net" / name
            (d / "device").mkdir(parents=True)
            if driver:
                os.symlink(driver, d / "device" / "driver")
            for f, v in (("operstate", state), ("speed", "200000"), ("mtu", "9000"), ("address", "a0:88:c2:12:34:56")):
                (d / f).write_text(v + "\n")
        port = root / "ib" / "mlx5_0" / "ports" / "1"
        port.mkdir(parents=True)
        for f, v in (("state", "4: ACTIVE"), ("rate", "200 Gb/sec (2X NDR)"), ("link_layer", "Ethernet")):
            (port / f).write_text(v + "\n")
        out = P.ports(net=str(root / "net"), ib=str(root / "ib"))
        self.assertEqual([p["name"] for p in out["ports"]], ["enp1s0f0np0"], "only the mlx5 ports")
        self.assertEqual(out["ports"][0]["speed_mbps"], "200000")
        self.assertEqual(out["rdma"], [{"device": "mlx5_0/1", "state": "4: ACTIVE", "rate": "200 Gb/sec (2X NDR)",
                                        "link": "Ethernet"}])
        self.assertNotRegex(json.dumps(out), MAC)


class WhatItNeverOpens(unittest.TestCase):
    def test_no_file_that_holds_a_key(self):
        opened = []
        real_open, real_read = builtins.open, pathlib.Path.read_text

        def spy_open(f, *a, **k):
            opened.append(str(f))
            return real_open(f, *a, **k)

        def spy_read(self, *a, **k):
            opened.append(str(self))
            return real_read(self, *a, **k)
        with mock.patch("builtins.open", spy_open), mock.patch.object(pathlib.Path, "read_text", spy_read), \
                mock.patch("sys.stdout", io.StringIO()):
            P.main(["--json"])
        cfg = str(P.CFG)
        self.assertEqual([f for f in opened if f.startswith(cfg)], [str(P.CFG / "installed.json")])
        self.assertFalse([f for f in opened if re.search(r"key|secret|token|\.env\b|password", f, re.I)], opened)


class OnThisBox(unittest.TestCase):
    def test_what_it_prints_names_nothing(self):
        for args in ([], ["--json"], ["--peer", "127.0.0.1"]):
            r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotRegex(r.stdout, MAC)
            self.assertNotRegex(r.stdout, IPV4)
            self.assertNotIn(P.HOME, r.stdout)
        facts = json.loads(subprocess.run([sys.executable, str(SCRIPT), "--json"], capture_output=True,
                                          text=True, timeout=120).stdout)
        self.assertEqual(sorted(facts), ["box", "connectx7", "repo", "software"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
