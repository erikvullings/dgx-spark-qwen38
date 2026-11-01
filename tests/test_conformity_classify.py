#!/usr/bin/env python3
"""What a box serves, read back from its units, and each difference from the recommended state
classed as a choice or as an old default (conformity.py, 2026-10-17).

The units here are rendered from the repo's own templates with the recommended values, then
one value at a time is moved: a box rendered that way must show no difference at all, and each
moved value must be read back the way install.sh reads it. Then the classing, case by case, as
the plan of v1.36.0 lists them: a value an older release installed by default is an old
default on a box that is older than the change, a choice on a younger one, a choice when the
age cannot be told; once a run has written installed.json, only what it records is a choice.
"""
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import installer_wall as wall  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import conformity  # noqa: E402
from unittest import mock  # noqa: E402

TEXT = (REPO / "install.sh").read_text()
HOME = "/home/u"
# conformity.py recommends by the box's class (the native window on a 64 GB one): these hold a
# 128 GB box's state, so the class is that one whatever the machine running them is, and an
# exported QWEN38_MEMORY_CLASS is back as it was after them (found in review, 2026-10-23).
_ENV = mock.patch.dict(os.environ, {"QWEN38_MEMORY_CLASS": "128"})


def setUpModule():
    _ENV.start()


def tearDownModule():
    _ENV.stop()


with _ENV:
    REC = conformity.recommended(home=HOME)
OLD = {o.knob: o for o in conformity.OLD_DEFAULTS}
ZLAB = ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4", "unquant", "8")
NOW = "2026-10-20T08:12:00+00:00"


def tier_args(tier, replay="1"):
    """FLASH_TIER_ARGS as install.sh's own resolve_flash_tier_args builds them."""
    start = TEXT.index("resolve_flash_tier_args() {")
    func = TEXT[start:TEXT.index("\n}\n", start) + 3]
    r = subprocess.run(["bash", "-c", f'set -eu\nFLASH_TIER={tier}; FLASH_REPLAYSSM_SPEC={replay}\n'
                        + func + 'resolve_flash_tier_args\necho "$FLASH_TIER_ARGS"\n'],
                       capture_output=True, text=True, timeout=30, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def ip_deny(bind):
    """The diffusion units' deny list for this bind, by lanes.sh's own rule (lane_ip_deny), not a copy of it."""
    r = subprocess.run(["bash", "-c", f'. "{REPO}/lanes.sh"; lane_ip_deny "$1"', "_", bind],
                       capture_output=True, text=True, timeout=30, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def render(template, values):
    text = (REPO / template).read_text()
    for k, v in values.items():
        text = text.replace(f"__{k}__", v)
    left = set(re.findall(r"__[A-Z0-9_]+__", text))
    assert not left, f"{template}: placeholders this fixture does not fill: {left}"
    return text


def box(context="1m", host="127.0.0.1", port="30000", hf=None, target27="RadixArk/Qwen3.8-27B-NVFP4",
        draft=("maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal", "bd7a934213c47a9e7ef69eef36bb3325f47fd1f1",
               "modelopt_fp4", "16"),
        flash=True, flash_target="RadixArk/Qwen3.8-Flash-Next-NVFP4", tier="concurrency", replay="1",
        token_map="65536", ple=None, proxy_bind="0.0.0.0", proxy_port="30001", key_at=None, record=None,
        side=True, kolibri="Aleph-Alpha/Kolibri-1", image_model="Qwen/Qwen-Image-2.1", video_variant="fl2va"):
    """(units dir, config dir) of a box whose units are the templates rendered with these values
    (the recommended ones by default). key_at: the mtime of its API key (None: no key);
    record: installed.json, as a dict or as raw text; side: the Kolibri, image and video units."""
    units, cfg = wall._tmp("units-"), wall._tmp("cfg-")
    hf = hf or f"{HOME}/.cache/huggingface"
    common = {"HOME": HOME, "USER": "u", "GROUP": "u", "PORT": port, "ENGINE_BIND": host, "HF_CACHE": hf,
              "IMAGE": "lmsysorg/sglang@sha256:" + "a" * 64, "KV_CACHE_ARGS": ""}
    tpl = "qwen38-sglang-1m.service.template" if context == "1m" else "qwen38-sglang.service.template"
    (units / "qwen38-sglang.service").write_text(render(tpl, {
        **common, "MODEL": target27, "MODEL_REV_ARGS": "--revision " + "b" * 40,
        "DRAFT2_REPO": draft[0], "DRAFT2_REV": draft[1], "DRAFT2_QUANT": draft[2], "DRAFT2_TOKENS": draft[3]}))
    if flash:
        args = tier_args(tier, replay)
        line = (f"TIER+=(--speculative-token-map /out/token-map-{token_map}.pt)"
                if "--speculative-algorithm" in args and token_map != "0" else "")
        (cfg / "launch-flash.sh").write_text(render("qwen38-flash-launch.sh.template", {
            **common, "MODEL": flash_target, "MODEL_REV_ARGS": "--revision " + "c" * 40,
            "PLE_DIR": ple or f"{HOME}/flashnext-ple", "FLASH_TIER_ARGS": args, "SPEC_TOKEN_MAP_LINE": line,
            "FLASH_QUANT_ARGS": "--quantization modelopt_fp4", "FLASH_MEM_FRACTION": "0.85",
            "PLE_RSS_BUDGET_GB": "8"}))
        (units / "qwen38-flash.service").write_text(render("qwen38-flash.service.template", common))
    (units / "qwen38-keepalive.service").write_text(render("qwen38-keepalive.service.template", {
        **common, "PROXY_PORT": proxy_port, "PROXY_BIND": proxy_bind, "PROMPT_CEILING": "0",
        "FLASH_PROMPT_CEILING": "250000"}))
    if side:
        (units / "qwen38-kolibri.service").write_text(render("qwen38-kolibri.service.template", {
            **common, "PORT": "30030", "GMU": "0.80", "IMAGE": "qwen38-kolibri:x", "MODEL": kolibri, "REV": "e" * 40}))
        lanes = f"{HOME}/.local/share"
        (units / "qwen38-image.service").write_text(render("qwen38-image.service.template", {
            **common, "IMAGE_BIND": host, "IMAGE_IP_DENY": ip_deny(host), "IMAGE_IP_ALLOW": "localhost" if ip_deny(host) else "", "IMAGE_PORT": "30010", "IMAGE_LANE_DIR": f"{lanes}/qwen38-image",
            "IMAGE_VENV": f"{lanes}/qwen38-image/venv", "IMAGE_MODEL": image_model}))
        (units / "qwen38-video.service").write_text(render("qwen38-video.service.template", {
            **common, "VIDEO_BIND": host, "VIDEO_IP_DENY": ip_deny(host), "VIDEO_IP_ALLOW": "localhost" if ip_deny(host) else "", "VIDEO_PORT": "30022", "VIDEO_LANE_DIR": f"{lanes}/qwen38-video",
            "VIDEO_VENV": f"{lanes}/qwen38-video/venv", "VIDEO_MODEL": "MiniMaxAI/MiniMax-H3",
            "VIDEO_VARIANT": video_variant}))
    if key_at is not None:
        (cfg / "api-key").write_text("not-a-key\n")
        t = datetime.datetime.fromisoformat(key_at).timestamp()
        os.utime(cfg / "api-key", (t, t))
    if record is not None:
        (cfg / "installed.json").write_text(record if isinstance(record, str) else json.dumps(record))
    return str(units), str(cfg)


def a_record(**choices):
    return {"schema": 1, "release": "v1.36.0", "commit": "d" * 40, "path": "full",
            "box_since": "2026-09-05T09:00:00+00:00", "box_since_from": "api-key",
            "choices": {k: {"value": v, "since": NOW, "how": "env"} for k, v in choices.items()}}


def classes(units, cfg, lane="27b"):
    """{knob: (class, why)} of every difference this box shows."""
    return {d["knob"]: (d["class"], d["why"]) for d in conformity.deviations(units, cfg, lane=lane, home=HOME)}


def shift(instant, minutes):
    return (datetime.datetime.fromisoformat(instant) + datetime.timedelta(minutes=minutes)).isoformat()


class WhatTheBoxServes(unittest.TestCase):
    def test_a_box_rendered_with_the_recommended_values_shows_no_difference(self):
        for lane in ("27b", "flash"):
            with self.subTest(lane=lane):
                units, cfg = box()
                self.assertEqual(conformity.observe(units, cfg, lane=lane), REC)
                self.assertEqual(classes(units, cfg, lane), {})

    def test_each_setting_is_read_back_the_way_install_sh_reads_it(self):
        cases = (
            ({"target27": "Qwen/Qwen3.8-27B-FP8"}, "target.27b", "fp8"),
            ({"target27": "someone/own-27b"}, "target.27b", "someone/own-27b"),
            ({"flash_target": "nvidia/Qwen3.8-Flash-Next-NVFP4"}, "target.flash", "flash-nvda"),
            ({"draft": ZLAB}, "DRAFT2", "z-lab/Qwen3.8-27B-DFlash2 (unquant, D=8)"),
            ({"context": "native"}, "CONTEXT_MODE", "native"),
            ({"host": "0.0.0.0"}, "ENGINE_BIND", "0.0.0.0"),
            ({"proxy_bind": "127.0.0.1"}, "PROXY_BIND", "127.0.0.1"),
            ({"port": "30010", "proxy_port": "30011"}, "PORT", "30010"),
            ({"proxy_port": "8080"}, "PROXY_PORT", "8080"),
            ({"hf": "/mnt/hf"}, "HF_CACHE", "/mnt/hf"),
            ({"ple": "/mnt/ple"}, "PLE_DIR", "/mnt/ple"),
            ({"tier": "context"}, "FLASH_TIER", "context"),
            ({"tier": "throughput"}, "FLASH_TIER", "throughput"),
            ({"token_map": "0"}, "SPEC_TOKEN_MAP_SIZE", "0"),
            ({"token_map": "32768"}, "SPEC_TOKEN_MAP_SIZE", "32768"),
            ({"replay": "0"}, "FLASH_REPLAYSSM_SPEC", "0"),
        )
        for kw, knob, want in cases:
            with self.subTest(**{knob: want}):
                units, cfg = box(**kw)
                self.assertEqual(conformity.observe(units, cfg, lane="27b")[knob], want)

    def test_the_bind_port_and_cache_are_the_serving_lanes(self):
        """Both text lanes carry them; the run that installs reads them from the lane it serves."""
        units, cfg = box()
        p = pathlib.Path(cfg) / "launch-flash.sh"
        p.write_text(p.read_text().replace("--host 127.0.0.1", "--host 0.0.0.0"))
        self.assertEqual(conformity.observe(units, cfg, lane="27b")["ENGINE_BIND"], "127.0.0.1")
        self.assertEqual(conformity.observe(units, cfg, lane="flash")["ENGINE_BIND"], "0.0.0.0")

    def test_a_tier_that_does_not_speculate_says_nothing_of_the_draft_knobs(self):
        units, cfg = box(tier="throughput")
        seen = conformity.observe(units, cfg, lane="27b")
        self.assertNotIn("SPEC_TOKEN_MAP_SIZE", seen)
        self.assertNotIn("FLASH_REPLAYSSM_SPEC", seen)

    def test_a_lane_that_is_not_installed_says_nothing(self):
        units, cfg = box(flash=False)
        seen = conformity.observe(units, cfg, lane="27b")
        self.assertFalse({"target.flash", "FLASH_TIER", "PLE_DIR"} & set(seen))
        (pathlib.Path(units) / "qwen38-sglang.service").unlink()
        self.assertFalse({"target.27b", "DRAFT2", "CONTEXT_MODE", "ENGINE_BIND"} & set(conformity.observe(units, cfg)))

    def test_the_proxy_port_follows_the_engine_port_it_is_recommended_beside(self):
        units, cfg = box(port="30010", proxy_port="30011")
        self.assertEqual(classes(units, cfg), {"PORT": ("choice", "site")})


class AnOldDefaultOnABoxWithoutInstalledJson(unittest.TestCase):
    """The first update to v1.36.0: nothing records what was chosen, so the age of the box
    decides, read from its API key (written once, at the first install)."""

    def test_older_than_the_change_it_is_an_old_default(self):
        for knob, kw in (("ENGINE_BIND", {"host": "0.0.0.0"}), ("CONTEXT_MODE", {"context": "native"}),
                         ("FLASH_TIER", {"tier": "context"}), ("DRAFT2", {"draft": ZLAB})):
            with self.subTest(knob=knob):
                units, cfg = box(key_at=shift(OLD[knob].instant, -60), **kw)
                self.assertEqual(classes(units, cfg)[knob], ("old-default", "box-before"))

    def test_younger_than_the_change_it_is_a_choice(self):
        for knob, kw in (("ENGINE_BIND", {"host": "0.0.0.0"}), ("CONTEXT_MODE", {"context": "native"}),
                         ("FLASH_TIER", {"tier": "context"}), ("DRAFT2", {"draft": ZLAB})):
            with self.subTest(knob=knob):
                units, cfg = box(key_at=shift(OLD[knob].instant, 60), **kw)
                self.assertEqual(classes(units, cfg)[knob], ("choice", "box-after"))

    def test_the_same_day_is_told_apart_by_the_instant(self):
        """v1.16.0 (engine on 0.0.0.0) was tagged at 13:01:26 and v1.17.0 at 13:19:42 the same
        day: a box installed from each is on its own side."""
        v1160 = "2026-09-22T13:01:26+02:00"
        units, cfg = box(host="0.0.0.0", key_at=v1160)
        self.assertEqual(classes(units, cfg)["ENGINE_BIND"][0], "old-default")
        units, cfg = box(host="0.0.0.0", key_at=OLD["ENGINE_BIND"].instant)
        self.assertEqual(classes(units, cfg)["ENGINE_BIND"][0], "choice", "the instant itself is after")
        units, cfg = box(host="0.0.0.0", key_at=shift(OLD["ENGINE_BIND"].instant, -1))
        self.assertEqual(classes(units, cfg)["ENGINE_BIND"][0], "old-default")

    def test_an_age_that_cannot_be_told_keeps_the_value(self):
        units, cfg = box(host="0.0.0.0", key_at=None)
        self.assertEqual(classes(units, cfg)["ENGINE_BIND"], ("choice", "box-unknown"))

    def test_a_key_restored_from_a_backup_reads_young_and_keeps(self):
        """cp gives the copy a new mtime: the box reads as installed today, so what it serves
        is kept and said, which is the safe way to be wrong."""
        units, cfg = box(host="0.0.0.0", context="native", key_at=NOW)
        self.assertEqual(classes(units, cfg), {"ENGINE_BIND": ("choice", "box-after"),
                                               "CONTEXT_MODE": ("choice", "box-after")})

    def test_an_unreadable_installed_json_is_no_record_and_says_so(self):
        for raw in ('{"schema": 1, "release": "v1.3', "not json", '[]', '{"schema": 2, "choices": {}}',
                    '{"schema": 1, "choices": []}', '{"schema": 1, "choices": {"ENGINE_BIND": "0.0.0.0"}}'):
            with self.subTest(raw=raw):
                units, cfg = box(host="0.0.0.0", key_at=shift(OLD["ENGINE_BIND"].instant, -60), record=raw)
                rec, note = conformity.load_record(cfg)
                self.assertIsNone(rec)
                self.assertIn("installed.json", note)
                self.assertEqual(classes(units, cfg)["ENGINE_BIND"], ("old-default", "box-before"))

    def test_a_lane_left_out_keeps_its_old_default_an_old_default(self):
        """A .off lane is not rendered by an update, so it still carries what it was installed
        with: the marker does not make that value a choice."""
        units, cfg = box(tier="context", key_at=shift(OLD["FLASH_TIER"].instant, -60))
        (pathlib.Path(cfg) / "flash.off").write_text("2026-09-30\n")
        self.assertEqual(classes(units, cfg)["FLASH_TIER"], ("old-default", "box-before"))


class OnceInstalledJsonExists(unittest.TestCase):
    """Every writer records an explicit choice since v1.36.0, so a value no entry names is an
    old default whatever the age of the box."""

    def test_a_recorded_choice_stays_a_choice(self):
        units, cfg = box(host="0.0.0.0", key_at=shift(OLD["ENGINE_BIND"].instant, -60),
                         record=a_record(ENGINE_BIND="0.0.0.0"))
        self.assertEqual(classes(units, cfg)["ENGINE_BIND"], ("choice", "recorded"))

    def test_an_unrecorded_old_default_is_an_old_default_even_on_a_young_box(self):
        units, cfg = box(host="0.0.0.0", key_at=NOW, record=a_record())
        self.assertEqual(classes(units, cfg)["ENGINE_BIND"], ("old-default", "not-recorded"))

    def test_an_entry_the_unit_no_longer_matches_is_set_aside(self):
        """Recorded throughput, served context (a hand edit, an old launcher): the entry says
        nothing about what is served, which is classed on its own."""
        units, cfg = box(tier="context", record=a_record(FLASH_TIER="throughput"))
        self.assertEqual(classes(units, cfg)["FLASH_TIER"], ("old-default", "not-recorded"))

    def test_a_value_no_release_ever_installed_is_a_choice_either_way(self):
        for rec in (None, a_record()):
            for kw, knob in (({"target27": "Qwen/Qwen3.8-27B-FP8"}, "target.27b"), ({"tier": "throughput"}, "FLASH_TIER"),
                             ({"draft": (*ZLAB[:3], "16")}, "DRAFT2"), ({"proxy_bind": "127.0.0.1"}, "PROXY_BIND")):
                with self.subTest(knob=knob, record=bool(rec)):
                    units, cfg = box(record=rec, key_at=shift(OLD["DRAFT2"].instant, -600), **kw)
                    self.assertEqual(classes(units, cfg)[knob], ("choice", "never-default"))

    def test_site_settings_are_choices_listed_apart(self):
        units, cfg = box(hf="/mnt/hf", ple="/mnt/ple", record=a_record())
        self.assertEqual(classes(units, cfg), {"HF_CACHE": ("choice", "site"), "PLE_DIR": ("choice", "site")})


class WhatARunRecords(unittest.TestCase):
    """next_choices: the entries a run writes, from what it was told and what the box serves."""

    def entries(self, units, cfg, chose=(), lane="27b"):
        rec, _ = conformity.load_record(cfg)
        seen = conformity.observe(units, cfg, lane=lane)
        since, _ = conformity.box_since(cfg, rec)
        return conformity.next_choices(rec, seen, conformity.recommended(home=HOME), dict(chose), since, NOW)

    def test_an_explicit_choice_is_recorded_as_such(self):
        units, cfg = box(host="0.0.0.0")
        self.assertEqual(self.entries(units, cfg, {"ENGINE_BIND": "0.0.0.0"})["ENGINE_BIND"],
                         {"value": "0.0.0.0", "since": NOW, "how": "env"})

    def test_passing_the_recommended_value_erases_the_entry(self):
        units, cfg = box(record=a_record(ENGINE_BIND="0.0.0.0"))
        self.assertEqual(self.entries(units, cfg, {"ENGINE_BIND": "127.0.0.1"}), {})

    def test_a_kept_choice_is_recorded_at_the_first_run_and_an_old_default_is_not(self):
        units, cfg = box(host="0.0.0.0", context="native", key_at="2026-09-20T10:00:00+02:00")
        got = self.entries(units, cfg)
        self.assertEqual(got, {"CONTEXT_MODE": {"value": "native", "since": NOW, "how": "kept"}},
                         "native on a box younger than v1.12.1 is a choice; 0.0.0.0 older than v1.17.0 is not")

    def test_an_entry_the_unit_contradicts_is_dropped(self):
        units, cfg = box(record=a_record(FLASH_TIER="throughput", PORT="30010"))
        self.assertEqual(self.entries(units, cfg), {})

    def test_an_explicit_choice_that_landed_nowhere_is_not_recorded(self):
        """FLASH_TIER=throughput on a box whose flash lane was not rendered by this run."""
        units, cfg = box(flash=False)
        self.assertEqual(self.entries(units, cfg, {"FLASH_TIER": "throughput"}), {})

    def test_an_entry_still_served_keeps_its_date(self):
        old = a_record(ENGINE_BIND="0.0.0.0")
        old["choices"]["ENGINE_BIND"]["since"] = "2026-10-18T00:00:00+00:00"
        units, cfg = box(host="0.0.0.0", record=old)
        self.assertEqual(self.entries(units, cfg)["ENGINE_BIND"]["since"], "2026-10-18T00:00:00+00:00")
        self.assertEqual(self.entries(units, cfg, {"ENGINE_BIND": "0.0.0.0"})["ENGINE_BIND"]["since"],
                         "2026-10-18T00:00:00+00:00")


class TheAgeOfTheBox(unittest.TestCase):
    def test_it_is_the_records_once_there_is_one_else_the_keys(self):
        units, cfg = box(key_at="2026-09-05T11:00:00+02:00")
        self.assertEqual(conformity.box_since(cfg, None), ("2026-09-05T09:00:00+00:00", "api-key"))
        self.assertEqual(conformity.box_since(cfg, a_record()), ("2026-09-05T09:00:00+00:00", "record"))
        units, cfg = box()
        self.assertEqual(conformity.box_since(cfg, None), (None, "unknown"))


class TheReleaseItRuns(unittest.TestCase):
    """Behind its own clone: a git pull with no ./install.sh, a last run stopped at --no-start,
    or a box no run of v1.36.0 or later has recorded."""

    def test_each_way_to_be_behind(self):
        head = "d" * 40
        self.assertEqual(conformity.release_lag(None, head), "no-record")
        self.assertEqual(conformity.release_lag(a_record(), "e" * 40), "commit")
        self.assertEqual(conformity.release_lag({**a_record(), "path": "no-start"}, head), "no-start")
        self.assertIsNone(conformity.release_lag(a_record(), head))
        self.assertIsNone(conformity.release_lag(a_record(), None), "a clone that cannot say its head is not behind")


if __name__ == "__main__":
    unittest.main(verbosity=2)
