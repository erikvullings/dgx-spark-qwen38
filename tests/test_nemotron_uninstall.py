#!/usr/bin/env python3
"""./install-nemotron.sh --uninstall gives the boot back to the box's text lane, takes its opencode
provider out, and keeps the key's middleware while the Kolibri lane uses it; and the Kolibri
lane's uninstall keeps it while the Nemotron lane does (vllm-auth is the vLLM lanes' one
folder: removed by either, it left the other's engine without its middleware, which its unit
mounts read-only and loads at every start). The Nemotron Super lane's uninstall does the same, and
the two Nemotron lanes run one image: each lane's uninstall keeps it while the other is installed.

The scripts' own uninstall blocks run here, whole, against a fake unit path, a fake config
directory and a fake /etc/systemd/system; sudo, systemctl and docker are stubs that write down
every call, as in test_kolibri_uninstall_chain.py.
"""
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


def block(script):
    text = (REPO / script).read_text()
    start = text.index('if [ "$ACTION" = uninstall ]; then')
    return text[start:text.index("\n  exit 0\nfi\n", start) + len("\n  exit 0\nfi\n")]


NEM_BLOCK = block("install-nemotron.sh")
KOL_BLOCK = block("install-kolibri.sh")
SUPER_BLOCK = block("install-nemotron-super.sh")
BLOCKS = {"nemotron": NEM_BLOCK, "kolibri": KOL_BLOCK, "nemotron-super": SUPER_BLOCK}
SCRIPTS = {"nemotron": "install-nemotron.sh", "kolibri": "install-kolibri.sh", "nemotron-super": "install-nemotron-super.sh"}


def vlane(script):
    """The installer's own names for what it shares with the other vLLM lanes (vllm-lane.sh)."""
    return next(ln for ln in (REPO / script).read_text().splitlines() if ln.startswith("VLANE="))
PINS = {"nemotron": ["NEMOTRON_BASE_IMAGE=vllm/vllm-openai@sha256:0",
                     "NEMOTRON_BOLT_REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
                     "NEMOTRON_BOLT_DRAFT_REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"],
        "kolibri": ["KOLIBRI_BASE_IMAGE=vllm/vllm-openai@sha256:0", "KOLIBRI_FP8_REPO=Aleph-Alpha/Kolibri-1",
                    "KOLIBRI_NVFP4_REPO=Quantinine/Kolibri-1-NVFP4A16-FP8"],
        "nemotron-super": ["NEMOTRON_BASE_IMAGE=vllm/vllm-openai@sha256:0",
                           "NEMOTRON_SUPER_REPO=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
                           "NEMOTRON_SUPER_DRAFT_REPO=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr"]}


class UninstallHarness(unittest.TestCase):
    """The scripts' uninstall blocks, run against fakes. No test of its own: the classes below
    declare each case once (the CI counts every test once against what unittest runs, and a class
    inheriting another's tests ran them twice under one name: test_lane_on_demand.py says so)."""

    def run_block(self, lane="nemotron", files=None, at_boot=False, opencode=None,
                  units=("qwen38-flash.service", "qwen38-sglang.service"), images=(), unit_texts=None,
                  sudo_fails=""):
        """lane: whose uninstall runs; units: what the fake /etc/systemd/system holds (unit_texts: what
        some of them say); images: the builds docker lists."""
        d = pathlib.Path(tempfile.mkdtemp(prefix="nem-uninstall-"))
        sd = d / "systemd"
        sd.mkdir()
        for u in units:
            (sd / u).write_text((unit_texts or {}).get(u, "[Service]\n"))
        self.home = d / "home"
        cfg = self.cfg = d / "cfg"
        cfg.mkdir()
        (self.home / ".config" / "opencode").mkdir(parents=True)
        (cfg / "vllm-auth").mkdir()
        (cfg / "nemotron-cache" / "vllm").mkdir(parents=True)
        (cfg / "nemotron-super-cache" / "vllm").mkdir(parents=True)
        (cfg / "vllm-auth" / "vllm_engine_auth.py").write_text("# the middleware\n")
        if opencode is not None:
            (cfg / "opencode.json").write_text(json.dumps(opencode, indent=2))
            (self.home / ".config" / "opencode" / "opencode.json").write_text(json.dumps(opencode, indent=2))
        for name, value in (files or {}).items():
            (cfg / name).write_text(value + "\n")
        bin_ = d / "bin"
        bin_.mkdir()
        log = self.log = d / "calls.log"
        (bin_ / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{log}"\n'
                                   + (f'case "$*" in *"{sudo_fails}"*) exit 1 ;; esac\n' if sudo_fails else "")
                                   + "exit 0\n")
        listed = "".join(f"echo {i}\n" for i in images)
        (bin_ / "docker").write_text(f'#!/bin/bash\necho "docker $*" >> "{log}"\n'
                                     f'[ "$1" = images ] && {{ :\n{listed}}}\nexit 0\n')
        unit_name = f"qwen38-{lane}.service"
        (bin_ / "systemctl").write_text(
            f'#!/bin/bash\necho "systemctl $*" >> "{log}"\n'
            f'[ "$1" = is-enabled ] && [ "${{@: -1}}" = {unit_name} ] && exit {0 if at_boot else 1}\n'
            '[ "$1" = is-enabled ] && exit 1\nexit 0\n')
        for f in bin_.iterdir():
            f.chmod(0o755)
        unit = d / unit_name
        unit.write_text("[Service]\n")
        pins = PINS[lane]
        script = "\n".join([
            "set -euo pipefail", "step(){ :; }",
            f'LANES_UNIT_DIR="{sd}"', f'. "{REPO}/lanes.sh"', f'. "{REPO}/vllm-lane.sh"', vlane(SCRIPTS[lane]),
            f'HERE="{REPO}"', "ACTION=uninstall", f"UNIT={unit_name}", f'INSTALLED="{unit}"',
            f"CONTAINER=qwen38-{lane}", f'CONFIG_DIR="{cfg}"', f'HF_CACHE="{d}/hf"', *pins,
            BLOCKS[lane].replace("/etc/systemd/system/", f"{sd}/"),
        ])
        env = dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin", HOME=str(self.home))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.out = r.stdout
        return cfg

    def enabled(self):
        return [ln.split()[-1] for ln in self.log.read_text().splitlines() if ln.startswith("sudo systemctl enable")]

    def conf(self, default):
        return {"$schema": "https://opencode.ai/config.json", "model": default, "small_model": default,
                "compaction": {"auto": True, "preserve_recent_tokens": 46000},
                "provider": {p: {"options": {"baseURL": "http://127.0.0.1:30001/v1", "apiKey": "{file:/k/api-key}"},
                                 "models": {m: {"name": m, "limit": {"context": c, "output": o}}}}
                             for p, m, c, o in (("qwen38", "qwen3.8-27b", 529000, 250000),
                                                ("flashnext", "qwen3.8-flash-next", 205000, 32000),
                                                ("kolibri", "kolibri-1", 228000, 32000),
                                                ("nemotron", "nemotron-3.5-bolt", 205000, 32000),
                                                ("nemotron-super", "nemotron-3.5-super-vl", 990000, 32000))}}


class TheUninstall(UninstallHarness):
    """The Nemotron (Bolt) lane's uninstall, and the cases each vLLM lane's shares, a subTest a lane."""

    def test_at_boot_it_gives_the_boot_back_to_the_text_lane_noted(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, at_boot=True)
        self.assertEqual(self.enabled(), ["qwen38-flash.service"])
        self.assertIn("sudo systemctl disable --now qwen38-nemotron.service", self.log.read_text())

    def test_with_nothing_noted_the_installed_text_lane_gets_the_boot(self):
        for lane in ("nemotron", "nemotron-super"):
            with self.subTest(lane=lane):
                self.run_block(lane=lane, at_boot=True)
                self.assertEqual(self.enabled(), ["qwen38-sglang.service"])
                self.run_block(lane=lane, at_boot=True, units=())
                self.assertEqual(self.enabled(), [])
                self.assertIn("no engine is enabled at boot any more", self.out)

    def test_not_at_boot_it_enables_nothing(self):
        for lane in ("nemotron", "nemotron-super"):
            with self.subTest(lane=lane):
                self.run_block(lane=lane, files={"text-lane": "qwen38-flash.service"}, at_boot=False)
                self.assertEqual(self.enabled(), [])

    def test_an_opencode_web_restart_that_fails_does_not_stop_the_uninstall(self):
        # its provider gone, opencode-web is restarted to read the change; a restart that fails (no
        # ticket, no terminal) left the uninstall half done: the shared step returned its failure, and
        # under set -e the installer stopped before the container, the images and the caches (found
        # in review, 2026-10-16). It is said, and the rest goes on.
        for lane, default in (("nemotron", "nemotron/nemotron-3.5-bolt"), ("kolibri", "kolibri/kolibri-1"),
                              ("nemotron-super", "nemotron-super/nemotron-3.5-super-vl")):
            with self.subTest(lane=lane):
                cfg = self.run_block(lane=lane, files={"text-lane": "qwen38-sglang.service"},
                                     opencode=self.conf(default), sudo_fails="systemctl restart opencode-web.service")
                calls = self.log.read_text()
                self.assertIn("sudo systemctl restart opencode-web.service", calls)
                self.assertIn(f"docker rm -f qwen38-{lane}", calls, "the uninstall went on")
                self.assertFalse((cfg / f"{lane}-cache").exists())
                self.assertIn("opencode-web.service could not restart", self.out)

    def test_opencode_loses_the_provider_and_its_default_follows_the_lane_before(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, opencode=self.conf("nemotron/nemotron-3.5-bolt"))
        for path in (self.cfg / "opencode.json", self.home / ".config/opencode/opencode.json"):
            left = json.loads(path.read_text())
            self.assertNotIn("nemotron", left["provider"], path)
            self.assertEqual((left["model"], left["small_model"]), ("flashnext/qwen3.8-flash-next",) * 2, path)

    def test_the_compaction_follows_the_default_put_back(self):
        self.run_block(files={"text-lane": "qwen38-sglang.service"}, opencode=self.conf("nemotron/nemotron-3.5-bolt"))
        want = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", "529000"],
                                  capture_output=True, text=True, check=True).stdout)
        left = json.loads((self.cfg / "opencode.json").read_text())
        self.assertEqual(left["model"], "qwen38/qwen3.8-27b")
        self.assertEqual(left["compaction"]["preserve_recent_tokens"], want)

    def test_a_default_that_was_not_nemotron_is_left_where_it_was(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, opencode=self.conf("qwen38/qwen3.8-27b"))
        left = json.loads((self.cfg / "opencode.json").read_text())
        self.assertEqual(left["model"], "qwen38/qwen3.8-27b")
        self.assertNotIn("nemotron", left["provider"])

    def test_the_middleware_stays_while_the_other_vllm_lane_uses_it(self):
        for lane, other in (("nemotron", "qwen38-kolibri.service"), ("kolibri", "qwen38-nemotron.service"),
                            ("nemotron", "qwen38-nemotron-super.service"), ("kolibri", "qwen38-nemotron-super.service"),
                            ("nemotron-super", "qwen38-kolibri.service"), ("nemotron-super", "qwen38-nemotron.service")):
            with self.subTest(lane=lane):
                cfg = self.run_block(lane=lane, units=("qwen38-sglang.service", other))
                self.assertTrue((cfg / "vllm-auth" / "vllm_engine_auth.py").exists(),
                                f"the {other} engine would start without its middleware")
                self.assertIn("vllm-auth kept", self.out)
                cfg = self.run_block(lane=lane, units=("qwen38-sglang.service",))
                self.assertFalse((cfg / "vllm-auth").exists(), "alone, it goes with the lane")

    def test_its_images_container_and_caches_go_and_its_checkpoints_stay(self):
        self.run_block()
        calls = self.log.read_text()
        self.assertIn("docker rm -f qwen38-nemotron", calls)
        self.assertIn("docker images --format {{.Repository}}:{{.Tag}} qwen38-nemotron", calls)
        self.assertFalse((self.cfg / "nemotron-cache").exists())
        self.assertTrue((self.cfg / "nemotron-super-cache").exists(), "the other lane's cache is its own")

    def test_the_one_image_stays_while_the_other_nemotron_lane_runs_it(self):
        # the two Nemotron lanes run one image: a build stays while an installed unit names it,
        # and goes with the last lane that does (lanes.sh, lane_prune_images)
        img = "qwen38-nemotron:build1"
        for lane, other in (("nemotron", "qwen38-nemotron-super.service"), ("nemotron-super", "qwen38-nemotron.service")):
            with self.subTest(lane=lane):
                self.run_block(lane=lane, units=("qwen38-sglang.service", other), images=(img,),
                               unit_texts={other: f"[Service]\nExecStart=/usr/bin/docker run {img} \\\n"})
                self.assertNotIn(f"docker rmi {img}", self.log.read_text())
                self.assertIn(f"image kept: {img}", self.out)
                self.run_block(lane=lane, units=("qwen38-sglang.service",), images=(img,))
                self.assertIn(f"docker rmi {img}", self.log.read_text())


class TheServiceStepPrunesOnlyWhatNoUnitNames(unittest.TestCase):
    """Each vLLM installer's service step (install_service, run on every install and update, in
    --units-only too) removes the builds of its image from older pins: never one that another
    installed lane's unit still names. Bolt's update removed the build Super VL's unit named, and
    Super VL's next Load looped on preparing one already built (found in review, 2026-10-16)."""

    def service_step(self, script, unit, image_repo, units):
        text = (REPO / script).read_text()
        start = text.index("install_service(){")
        body = text[start:text.index("\n}\n", start) + 3]
        d = pathlib.Path(tempfile.mkdtemp(prefix="vllm-service-"))
        sd = d / "systemd"
        sd.mkdir()
        for name, says in units.items():
            (sd / name).write_text(says)
        bin_ = d / "bin"
        bin_.mkdir()
        log = d / "calls.log"
        (bin_ / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{log}"\nexit 0\n')
        (bin_ / "systemctl").write_text(f'#!/bin/bash\necho "systemctl $*" >> "{log}"\nexit 0\n')
        (bin_ / "docker").write_text(f'#!/bin/bash\necho "docker $*" >> "{log}"\n'
                                     f'[ "$1" = images ] && {{ echo {image_repo}:new; echo {image_repo}:old; }}\nexit 0\n')
        for f in bin_.iterdir():
            f.chmod(0o755)
        script_text = "\n".join([
            "set -euo pipefail", "step(){ :; }", "die(){ echo \"DIE $*\"; exit 1; }",
            f'LANES_UNIT_DIR="{sd}"', f'. "{REPO}/lanes.sh"', f'. "{REPO}/vllm-lane.sh"', vlane(script),
            f'HERE="{REPO}"', f"UNIT={unit}", f'INSTALLED="{sd}/{unit}"', f'CONFIG_DIR="{d}/cfg"',
            f"LOCAL_IMAGE={image_repo}:new", "MODEL=org/m", f"MODEL_REV={'a' * 40}", "DRAFT=org/d",
            f"DRAFT_REV={'b' * 40}", "GMU=0.7", "SEQS=4", f'HF_CACHE="{d}/hf"', "ENGINE_BIND=127.0.0.1",
            "PORT=30000", "SMOKE=1", body, "install_service",
        ])
        r = subprocess.run(["bash", "-c", script_text], capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin", HOME=str(d)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return log.read_text(), r.stdout

    def test_a_nemotron_lane_keeps_the_build_the_other_lanes_unit_names(self):
        for script, unit, other in (("install-nemotron.sh", "qwen38-nemotron.service", "qwen38-nemotron-super.service"),
                                    ("install-nemotron-super.sh", "qwen38-nemotron-super.service", "qwen38-nemotron.service")):
            with self.subTest(script=script):
                calls, out = self.service_step(script, unit, "qwen38-nemotron",
                                               {other: "ExecStart=/usr/bin/docker run qwen38-nemotron:old \\\n"})
                self.assertNotIn("docker rmi qwen38-nemotron:old", calls, out)
                self.assertIn("image kept: qwen38-nemotron:old", out)
                self.assertNotIn("docker rmi qwen38-nemotron:new", calls)
                # no unit names it any more: it goes, from either lane's update
                calls, out = self.service_step(script, unit, "qwen38-nemotron", {})
                self.assertIn("docker rmi qwen38-nemotron:old", calls)
                self.assertNotIn("docker rmi qwen38-nemotron:new", calls)

    def test_kolibri_still_removes_its_older_builds(self):
        calls, _ = self.service_step("install-kolibri.sh", "qwen38-kolibri.service", "qwen38-kolibri", {})
        self.assertIn("docker rmi qwen38-kolibri:old", calls)
        self.assertNotIn("docker rmi qwen38-kolibri:new", calls)


class TheSuperVLUninstall(UninstallHarness):
    """The Nemotron Super lane's uninstall, the cases that are its own: its unit and its provider
    (those it shares with Bolt's are TheUninstall's, a subTest a lane)."""

    def run_block(self, lane="nemotron-super", **kw):
        return super().run_block(lane=lane, **kw)

    def test_at_boot_it_gives_the_boot_back_to_the_text_lane_noted(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, at_boot=True)
        self.assertEqual(self.enabled(), ["qwen38-flash.service"])
        self.assertIn("sudo systemctl disable --now qwen38-nemotron-super.service", self.log.read_text())

    def test_opencode_loses_the_provider_and_its_default_follows_the_lane_before(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"},
                       opencode=self.conf("nemotron-super/nemotron-3.5-super-vl"))
        for path in (self.cfg / "opencode.json", self.home / ".config/opencode/opencode.json"):
            left = json.loads(path.read_text())
            self.assertNotIn("nemotron-super", left["provider"], path)
            self.assertIn("nemotron", left["provider"], "Bolt's provider is its own")
            self.assertEqual((left["model"], left["small_model"]), ("flashnext/qwen3.8-flash-next",) * 2, path)

    def test_the_compaction_follows_the_default_put_back(self):
        self.run_block(files={"text-lane": "qwen38-sglang.service"},
                       opencode=self.conf("nemotron-super/nemotron-3.5-super-vl"))
        want = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", "529000"],
                                  capture_output=True, text=True, check=True).stdout)
        left = json.loads((self.cfg / "opencode.json").read_text())
        self.assertEqual(left["model"], "qwen38/qwen3.8-27b")
        self.assertEqual(left["compaction"]["preserve_recent_tokens"], want)

    def test_a_default_that_was_not_nemotron_is_left_where_it_was(self):
        self.run_block(files={"text-lane": "qwen38-flash.service"}, opencode=self.conf("nemotron/nemotron-3.5-bolt"))
        left = json.loads((self.cfg / "opencode.json").read_text())
        self.assertEqual(left["model"], "nemotron/nemotron-3.5-bolt")
        self.assertNotIn("nemotron-super", left["provider"])

    def test_its_images_container_and_caches_go_and_its_checkpoints_stay(self):
        self.run_block()
        calls = self.log.read_text()
        self.assertIn("docker rm -f qwen38-nemotron-super", calls)
        self.assertIn("docker images --format {{.Repository}}:{{.Tag}} qwen38-nemotron", calls)
        self.assertFalse((self.cfg / "nemotron-super-cache").exists())
        self.assertTrue((self.cfg / "nemotron-cache").exists(), "Bolt's cache is its own")


if __name__ == "__main__":
    unittest.main()
