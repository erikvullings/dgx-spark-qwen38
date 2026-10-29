#!/usr/bin/env python3
"""pair-probe.py, the read-only probe a pair of Spark boxes runs before a dual-box lane (#45).

What it prints is pasted into a public discussion, so the invariant is what it never prints: a MAC
address, an IP address (only its kind), the home directory, the host's name, the box's serial
number, or anything of a file that holds a key, which it never opens. The cases after the first
seven come from the reviews of 2026-10-28 and 29: a peer that does not answer read as not measured,
the checkout was always ~/dgx-spark-qwen38, a lane systemd had unloaded was left out, the modules told
nothing, the model cache was the default one, a peer's name was not resolved, an IPv6 address went
through, a host named like a key renamed it, a record of another shape raised, a scoped IPv6 peer lost
its port, a name in another case, the user's name, a name next to a dash and a port named after its MAC
went through, a short host name ate the facts it names, and a ping in another language read nothing."""
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


def ipv6_in(text):
    """The valid IPv6 addresses written in text."""
    found = []
    for m in re.finditer(r"(?<![\w.])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])", text, re.I):
        try:
            found.append(str(__import__("ipaddress").IPv6Address(m.group(0))))
        except ValueError:
            pass
    return found


def tmpdir(case):
    """A directory that goes with the test."""
    d = tempfile.TemporaryDirectory()
    case.addCleanup(d.cleanup)
    return pathlib.Path(d.name)


def commands(answers):
    """A stand-in for subprocess.run: what each command (by its first two words) prints and ends with,
    and the list of what was run."""
    ran = []

    def fake(argv, **kw):
        ran.append(list(argv))
        for prefix, (rc, out) in answers.items():
            if " ".join(argv).startswith(prefix):
                return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="not this one")
    return fake, ran


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
        # in any case since the second review: a name is masked wherever it shows, this repo's name aside
        out = P.masked("dgx-spark-qwen38 on qwen38, a Qwen38 box", host="qwen38", user="nobody-here")
        self.assertEqual(out, "dgx-spark-qwen38 on <host>, a <host> box")

    def test_a_name_every_such_box_prints_stays(self):
        # a host "spark" of a user "nvidia" read "<user> DGX <host>" and "6.17.0-1032-<user>": no clue to
        # whose box it is, and the facts the probe compares were gone (review, 2026-10-29)
        box = {"machine": "NVIDIA DGX Spark", "kernel": "6.17.0-1032-nvidia", "os": "Ubuntu 24.04.3 LTS",
               "gpu": "NVIDIA GB10, 580.95.05"}
        self.assertEqual(P.masked_facts({"box": box}, host="spark", user="nvidia")["box"], box)
        box = {"machine": "ASUSTeK COMPUTER INC. GX10", "os": "Ubuntu 24.04.3 LTS"}
        self.assertEqual(P.masked_facts({"box": box}, host="GX10", user="ubuntu")["box"], box)
        self.assertEqual(P.masked("6.17.0-1032-nvidia on hasan-box", host="hasan-box", user="hasan"),
                         "6.17.0-1032-nvidia on <host>", "the owner's own name still goes")

    def test_a_version_of_five_parts_is_no_address(self):
        # the four first parts of a longer dotted number read as an address (review, 2026-10-29)
        self.assertEqual(P.masked("fw 1.2.3.4.5, build 7.6.0.1.2, at 10.0.0.1., net 10.0.0.0/8", host="", user=""),
                         "fw 1.2.3.4.5, build 7.6.0.1.2, at <private>., net <private>")

    def test_a_name_in_another_case_and_the_users_go(self):
        facts = {"box": {"kernel": "6.17.0-GX10-EFF9", "machine": "ASUSTeK GX10"}, "repo": {"branch": "Hasan/pairs"}}
        out = P.masked_facts(facts, host="gx10-eff9", user="hasan")
        self.assertEqual((out["box"]["kernel"], out["repo"]["branch"]), ("6.17.0-<host>", "<user>/pairs"))
        self.assertEqual(out["box"]["machine"], "ASUSTeK GX10", "a word of the name is not the name")

    def test_a_name_of_letters_next_to_a_dash_goes(self):
        # the example the code's own comment gives, and a branch named after its author (review, 2026-10-29)
        facts = {"box": {"kernel": "6.17.0-atlas-b"}, "repo": {"branch": "fix-hasan"},
                 "software": {"docker": "dgx-spark-qwen38 on DGX-Spark-Qwen38"}}
        out = P.masked_facts(facts, host="atlas", user="hasan")
        self.assertEqual((out["box"]["kernel"], out["repo"]["branch"]), ("6.17.0-<host>-b", "fix-<user>"))
        self.assertEqual(P.masked("dgx-spark-qwen38 on DGX-Spark-Qwen38", host="qwen38", user=""),
                         "dgx-spark-qwen38 on DGX-Spark-Qwen38", "this repo's own name, in any case")

    def test_a_port_named_after_its_mac_goes(self):
        self.assertEqual(P.masked("via enx00e04c680a1b and enx-not-a-mac", host="", user=""),
                         "via enx<mac> and enx-not-a-mac")

    def test_a_short_host_name_leaves_the_facts_it_names(self):
        facts = {"peer": {"via": "lo", "kind": "loopback"}, "software": {"time_synced": "yes"},
                 "connectx7": {"ports": [{"name": "enp1s0f0np0", "state": "up"}]}, "box": {"os": "lo up yes"}}
        out = P.masked_facts(facts, host="lo", user="yes")
        self.assertEqual((out["peer"]["via"], out["software"]["time_synced"], out["connectx7"]["ports"][0]["state"]),
                         ("lo", "yes", "up"), "words of the kernel and of systemd")
        self.assertEqual(out["box"]["os"], "<host> up <user>", "free text is masked")

    def test_an_address_after_a_colon_or_with_zeros_is_masked(self):
        out = P.masked("inet6 addr:fe80::a088:c2ff:fe12:3456/64 Scope:Link, inet 192.168.001.002", host="", user="")
        self.assertEqual(out, "inet6 addr:<link-local> Scope:Link, inet <private>")

    def test_an_ipv6_address_is_only_its_kind(self):
        out = P.masked("global 2a01:e0a:1f:2::5/64 link fe80::1%enp1s0f0np0 lo ::1, at 12:34:56", host="")
        self.assertEqual(out, "global <public> link <link-local> lo <loopback>, at 12:34:56")

    def test_a_host_named_like_a_key_renames_nothing(self):
        facts = {"box": {"gpu": "NVIDIA GB10", "machine": "ASUS on gpu"},
                 "connectx7": {"ports": [{"name": "enp1s0f0np0", "note": "link/ether a0:88:c2:12:34:56"}]}}
        with mock.patch.object(P.socket, "gethostname", return_value="gpu"):
            text = P.report(facts)
            with mock.patch("sys.stdout", io.StringIO()) as out:
                P.main(["--json"])
        self.assertIn("\n  gpu: NVIDIA GB10\n", text)
        self.assertIn("ASUS on <host>", text)
        self.assertNotRegex(text, MAC, "a value inside a list is masked as well")
        self.assertIn("gpu", json.loads(out.getvalue())["box"])


class TheBox(unittest.TestCase):
    def test_the_version_is_the_last_update_and_the_serial_is_never_read_out(self):
        rel = tmpdir(self) / "dgx-release"
        # the keys DGX OS writes, DGX_SERIAL_NUMBER among them (read on the reference box, values aside)
        rel.write_text('DGX_NAME="DGX Spark"\nDGX_PRETTY_NAME="NVIDIA DGX Spark"\nDGX_SWBUILD_DATE="2025-09-10"\n'
                       'DGX_SWBUILD_VERSION="7.2.3"\nDGX_COMMIT_ID="0123abc"\nDGX_PLATFORM="GX10"\n'
                       'DGX_SERIAL_NUMBER="SN-1234567890"\n\nDGX_OTA_VERSION="7.3.1"\nDGX_OTA_DATE="x"\n\n'
                       'DGX_OTA_VERSION="7.6.0"\nDGX_OTA_DATE="y"\n')
        facts = P.box(release=str(rel))
        self.assertEqual((facts["dgx_os"], facts["platform"]), ("7.6.0", "GX10"))
        self.assertNotIn("SN-1234567890", P.report({"box": facts}))
        self.assertNotIn("SN-1234567890", json.dumps(facts))


class ThePorts(unittest.TestCase):
    def test_the_connectx7_ports_and_their_rdma_side(self):
        root = tmpdir(self)
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

    def test_the_modules_are_said_when_no_card_shows(self):
        # the reference GX10: mlx5_core and mlx5_ib loaded, no card in lspci, no mlx5 port, no RDMA device
        root = tmpdir(self)
        for name in ("net", "ib", "module/mlx5_core", "module/mlx5_ib", "module/nvidia"):
            (root / name).mkdir(parents=True)
        out = P.ports(net=str(root / "net"), ib=str(root / "ib"), modules=str(root / "module"))
        self.assertEqual((out["ports"], out["rdma"], out["mlx5_modules"]), ([], [], ["mlx5_core", "mlx5_ib"]))
        with mock.patch.object(P.shutil, "which", return_value=None):
            out = P.ports(net=str(root / "net"), ib=str(root / "ib"), modules=str(root / "net"))
        self.assertEqual((out["mlx5_modules"], out["lspci"]), ([], "missing"))


class ThePeer(unittest.TestCase):
    def test_a_peer_that_does_not_answer_is_all_lost(self):
        # iputils ping ends 1 when no reply came: the down link is the case this check is for
        fake, _ = commands({"ip route": (0, "192.0.2.1 via 192.0.2.254 dev enp1s0f0np0 src 192.0.2.9 uid 1000\n"),
                            "ping": (1, "PING 192.0.2.1 (192.0.2.1) 56(84) bytes of data.\n\n--- 192.0.2.1 ping "
                                        "statistics ---\n4 packets transmitted, 0 received, 100% packet loss, "
                                        "time 3061ms\n\n")})
        with mock.patch.object(P.subprocess, "run", fake):
            got = P.peer("192.0.2.1")
        self.assertEqual((got["via"], got["loss_pct"], got["rtt_ms"]), ("enp1s0f0np0", 100.0, None))

    def test_a_scoped_ipv6_peer_keeps_its_port(self):
        # a point-to-point ConnectX-7 link with no IPv4 has link-local addresses only, scoped by their port
        fake, ran = commands({"ping": (0, "--- fe80::1%lo ping statistics ---\n4 packets transmitted, 4 received, "
                                          "0% packet loss, time 3ms\nrtt min/avg/max/mdev = 0.010/0.020/0.030/0.010 ms\n")})
        with mock.patch.object(P.subprocess, "run", fake):
            got = P.peer("fe80::1%lo")
        self.assertEqual((got["kind"], got["via"], got["loss_pct"], got["rtt_ms"]), ("link-local", "lo", 0.0, 0.02))
        self.assertIn(["ping", "-c", "4", "-W", "1", "fe80::1%lo"], ran)
        self.assertFalse([a for a in ran if a[:3] == ["ip", "route", "get"]], "ip route get refuses a scoped address")

    def test_commands_run_in_the_c_locale(self):
        seen = []

        def fake(argv, **kw):
            seen.append((kw.get("env") or {}).get("LC_ALL"))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        with mock.patch.object(P.subprocess, "run", fake):
            P.run(["true"])
        self.assertEqual(seen, ["C"], "what is read out of a command is English")

    def test_a_name_is_resolved_and_a_dash_is_no_address(self):
        got = P.peer("localhost")
        self.assertEqual((got["kind"], got["via"]), ("loopback", "lo"), "ip route get takes an address only")
        fake, ran = commands({})
        with mock.patch.object(P.subprocess, "run", fake):
            got = P.peer("-I")
        self.assertEqual((got["error"], ran), ("not an address", []), "ping would take it for an option")


class TheRepo(unittest.TestCase):
    def record(self, text):
        cfg = tmpdir(self)
        (cfg / "installed.json").write_text(text)
        return mock.patch.object(P, "CFG", cfg)

    def test_the_checkout_is_the_clone_the_box_was_installed_from(self):
        clone = tmpdir(self) / "elsewhere"
        git = ["git", "-C", str(clone), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
               "-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q", "-b", "probe-branch", str(clone)], check=True)
        subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "t"], check=True)
        head = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        with self.record(json.dumps({"release": "v9.9.9", "commit": "abcdef0123", "lane": "27b",
                                     "repo_dir": str(clone)})):
            got = P.repo()
        self.assertEqual((got["release"], got["commit"], got["checkout"], got["branch"]),
                         ("v9.9.9", "abcdef0", head, "probe-branch"))

    def test_a_unit_systemd_did_not_load_is_listed(self):
        fake, _ = commands({"systemctl list-units": (0, "qwen38-dashboard.service loaded active running Cockpit\n"),
                            "systemctl list-unit-files": (0, "qwen38-dashboard.service enabled enabled\n"
                                                             "qwen38-video.service disabled enabled\n")})
        with self.record("{}"), mock.patch.object(P.subprocess, "run", fake):
            got = P.repo()
        self.assertEqual(got["units"], {"qwen38-dashboard": "active/running (boot)", "qwen38-video": "inactive/dead"})

    def test_a_record_of_another_shape_is_read_as_none(self):
        for text in ("null", "[]", '"x"', '{"commit": 5, "release": ["v1"], "repo_dir": 3}'):
            with self.subTest(text=text), self.record(text):
                got = P.repo()
                self.assertEqual((got["release"], got["commit"], got["lane"]), ("", "", ""))


class TheSoftware(unittest.TestCase):
    def test_the_model_cache_is_the_one_the_lanes_mount(self):
        units = tmpdir(self)
        self.assertEqual(P.hf_cache(units), str(pathlib.Path(P.HOME) / ".cache" / "huggingface"), "no 27B unit")
        (units / "qwen38-sglang.service").write_text(
            "ExecStart=/usr/bin/docker run --rm -v /mnt/nvme//hf:/root/.cache/huggingface -v /x:/y img\n")
        asked = []
        with mock.patch.object(P, "_free", lambda path: asked.append(str(path))):
            P.software(units)
        self.assertEqual(asked, [P.HOME, "/mnt/nvme/hf"], "install.sh's HF_CACHE=, one slash at a time")

    def test_the_model_cache_of_a_box_without_the_27b(self):
        units = tmpdir(self)
        (units / "qwen38-flash-local.service").write_text(
            "ExecStart=/usr/bin/docker run -v /home/x/flash-local-ple/abc:/root/.cache/huggingface img\n")
        (units / "qwen38-kolibri.service").write_text(
            "ExecStart=/usr/bin/docker run -v /mnt/models:/root/.cache/huggingface img\n")
        self.assertEqual(P.hf_cache(units), "/mnt/models", "Flash FP4-Local mounts copies, never the cache")


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


SERIAL = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', P.read("/etc/dgx-release"), re.M)).get("DGX_SERIAL_NUMBER", "")


class OnThisBox(unittest.TestCase):
    def test_what_it_prints_names_nothing(self):
        for args in ([], ["--json"], ["--peer", "127.0.0.1"]):
            r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotRegex(r.stdout, MAC)
            self.assertNotRegex(r.stdout, IPV4)
            self.assertNotIn(P.HOME, r.stdout)
            self.assertEqual(ipv6_in(r.stdout), [])
            # said without the value, which a failure would print
            for what, value in (("the host's name", P.socket.gethostname()), ("the user's name", P.user_name()),
                                ("the serial number", SERIAL)):
                self.assertFalse(value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", r.stdout, re.I),
                                 f"{what} is in what it prints with {args}")
        facts = json.loads(subprocess.run([sys.executable, str(SCRIPT), "--json"], capture_output=True,
                                          text=True, timeout=120).stdout)
        self.assertEqual(sorted(facts), ["box", "connectx7", "repo", "software"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
