#!/usr/bin/env python3
"""The text lane the box comes back to: one record (lanes.sh), since v1.30.

Until v1.29 each side lane kept a link to the lane it took the boot from, written by every
switch, read by install.sh's six resolution lines and by three uninstalls, each its own way: a
round trip between side lanes made two links name each other (the reference box, 2026-10-14),
a side lane removed left another's link naming a unit that was gone, and the review of
2026-10-14 asked for one record. These run lanes.sh itself, then every sequence of switches
up to six long through switch-model.sh's own lines (each side lane's note, a text target's
note), then install.sh's own resolution and the uninstalls' own give-back, and hold them to
the one truth: the text lane is the last SGLang text lane the box served.
"""
import itertools
import os
import pathlib
import re
import subprocess
import tempfile
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import text_lane_fixture as fixture  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
LANES = REPO / "lanes.sh"
SWITCH = (REPO / "switch-model.sh").read_text()
INSTALL = (REPO / "install.sh").read_text()
UNIT = {"27b": "qwen38-sglang.service", "flash": "qwen38-flash.service", "image": "qwen38-image.service",
        "video": "qwen38-video.service", "kolibri": "qwen38-kolibri.service", "nemotron": "qwen38-nemotron.service",
        "nemotron-super": "qwen38-nemotron-super.service"}
SIDES = ("image", "video", "kolibri", "nemotron", "nemotron-super")


def side_note(lane):
    """switch-model.sh's lines that note the text lane when it switches to this side lane."""
    head = {"image": 'if [ "$CHOICE" = "image" ] || [ "$CHOICE" = "image-turbo" ]; then',
            "video": 'if [ "$CHOICE" = "video" ]; then',
            "kolibri": 'if [ "$CHOICE" = "kolibri" ] || [ "$CHOICE" = "kolibri-nvfp4" ]; then',
            # the two Nemotron lanes' one block
            "nemotron": 'if [ "$CHOICE" = "nemotron" ] || [ "$CHOICE" = "nemotron-super" ]; then',
            "nemotron-super": 'if [ "$CHOICE" = "nemotron" ] || [ "$CHOICE" = "nemotron-super" ]; then'}[lane]
    at = SWITCH.index(head)
    m = re.compile(r"\n  for TEXT_UNIT_NAME in [^\n]*; do\n    if systemctl is-enabled --quiet \"\$TEXT_UNIT_NAME\" "
                   r"2>/dev/null; then\n      text_lane_note \"\$TEXT_UNIT_NAME\"(; break[^\n]*)?\n    fi\n  done\n").search(SWITCH, at)
    assert m, f"switch-model.sh's note for the {lane} lane moved"
    return m.group(0)


NOTES = {lane: side_note(lane) for lane in SIDES}
assert 'text_lane_note "$TARGET_UNIT_NAME"' in SWITCH, "a switch to a text target no longer notes it"
_r = INSTALL.index("text_lane_migrate\nTEXT_LANE_LAST=")
RESOLVE = INSTALL[_r:INSTALL.index("\n", INSTALL.index("TEXT_LANE_LAST=", _r)) + 1]


class Box:
    """A config directory, a units directory and a systemctl that answers which unit is
    enabled at boot from a file. Its text lanes can start (text_lane_fixture) but those named
    in unready, whose unit is there and names nothing; sudo writes down what it is given."""

    def __init__(self, units=("27b", "flash", "image", "video", "kolibri", "nemotron", "nemotron-super"), unready=()):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="text-lane-"))
        self.cfg, self.units, self.bin = self.d / "cfg", self.d / "units", self.d / "bin"
        for p in (self.cfg, self.units, self.bin):
            p.mkdir()
        ready = [lane for lane in units if lane in ("27b", "flash") and lane not in unready]
        for lane in units:
            if lane not in ready:
                (self.units / UNIT[lane]).write_text("[Service]\n")
        fixture.put_ready(self.units, self.cfg, self.d / "hf", ready)
        (self.d / "enabled").write_text("")
        (self.bin / "systemctl").write_text(
            f'#!/bin/bash\n[ "$1" = is-enabled ] && grep -qx -- "${{@: -1}}" "{self.d}/enabled"\n')
        (self.bin / "docker").write_text("#!/bin/bash\n" + fixture.DOCKER_HAS_IT + "exit 1\n")
        (self.bin / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{self.d}/sudo"\n')
        for f in self.bin.iterdir():
            f.chmod(0o755)

    def run(self, body):
        script = (f'set -euo pipefail\nCONFIG_DIR="{self.cfg}"; LANES_UNIT_DIR="{self.units}"\n'
                  f'. "{LANES}"\n' + body)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.d)})
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout.strip()

    def boot(self, lane):
        (self.d / "enabled").write_text(UNIT[lane] + "\n")

    def files(self, **links):
        for name, value in links.items():
            (self.cfg / name.replace("_", "-")).write_text(value + "\n")
        return self


class TheLibrary(unittest.TestCase):
    def test_only_an_sglang_text_unit_is_noted(self):
        b = Box()
        for unit in ("qwen38-image.service", "qwen38-kolibri.service", "qwen38-nemotron.service",
                     "qwen38-nemotron-super.service", "qwen38-video.service", "x"):
            b.run(f'text_lane_note "{unit}"')
            self.assertFalse((b.cfg / "text-lane").exists(), unit)
        b.run('text_lane_note qwen38-flash.service')
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-flash.service\n")

    def test_the_record_wins_over_any_link(self):
        b = Box().files(text_lane="qwen38-flash.service", lane_before_video="qwen38-sglang.service")
        self.assertEqual(b.run("text_lane_last video"), "qwen38-flash.service")

    def test_the_links_are_followed_from_the_side_lane_at_boot_first(self):
        # the reference box before v1.30: image named Kolibri, Kolibri the flash, and an older
        # round trip left video naming the 27B
        b = Box().files(lane_before_image="qwen38-kolibri.service", lane_before_kolibri="qwen38-flash.service",
                        lane_before_video="qwen38-sglang.service")
        self.assertEqual(b.run("text_lane_last kolibri"), "qwen38-flash.service")
        self.assertEqual(b.run("text_lane_last image"), "qwen38-flash.service")
        self.assertEqual(b.run("text_lane_last video"), "qwen38-sglang.service")

    def test_a_chain_through_every_side_lane_is_followed(self):
        # video named Kolibri, Kolibri the image lane, the image lane the flash: three links, each
        # side lane once, and the same when they are read again after a record older than them
        b = Box().files(lane_before_video="qwen38-kolibri.service", lane_before_kolibri="qwen38-image.service",
                        lane_before_image="qwen38-flash.service")
        self.assertEqual(b.run("text_lane_last video"), "qwen38-flash.service")
        b = TheLibrary.dated(("text-lane", "qwen38-sglang.service", 400), ("lane-before-image", "qwen38-flash.service", 300),
                             ("lane-before-kolibri", "qwen38-image.service", 200),
                             ("lane-before-video", "qwen38-kolibri.service", 100))
        b.boot("video")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-flash.service\n")

    def test_links_that_loop_or_lead_nowhere_say_nothing(self):
        b = Box().files(lane_before_image="qwen38-kolibri.service", lane_before_kolibri="qwen38-image.service")
        self.assertEqual(b.run("text_lane_last image"), "")
        self.assertEqual(Box().run("text_lane_last"), "")

    def test_the_migration_writes_once_from_what_boots(self):
        b = Box().files(lane_before_image="qwen38-kolibri.service", lane_before_kolibri="qwen38-flash.service",
                        lane_before_video="qwen38-sglang.service")
        b.boot("kolibri")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-flash.service\n")
        b.boot("video")                       # later runs never read the links again
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-flash.service\n")

    def test_both_text_lanes_at_boot_note_the_27b(self):
        # install.sh follows the 27B first when both are enabled, and so does the note: it noted
        # the flash, the last one the loop met (found in review, 2026-10-14)
        for lane in SIDES:
            with self.subTest(lane=lane):
                b = Box()
                (b.d / "enabled").write_text("qwen38-sglang.service\nqwen38-flash.service\n")
                b.run(NOTES[lane])
                self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-sglang.service\n")

    def test_links_newer_than_the_record_are_read_again(self):
        # the box went back to v1.29, which switched from the 27B to the image lane and wrote its
        # link, then came to v1.30 again: the record says what v1.30 saw before (found in review,
        # 2026-10-14)
        b = Box().files(text_lane="qwen38-flash.service", lane_before_image="qwen38-sglang.service")
        now = time.time()
        os.utime(b.cfg / "text-lane", (now - 100, now - 100))
        os.utime(b.cfg / "lane-before-image", (now - 50, now - 50))
        b.boot("image")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-sglang.service\n")
        self.assertEqual(b.run("text_lane_back image"), "qwen38-sglang.service")

    def test_a_newer_link_through_an_older_one_keeps_the_record(self):
        # v1.29 switched the 27B to images (an old link), v1.30 noted the flash before going to
        # images again, then v1.29 went from images to video: the video lane's link is newer than
        # the record and leads to the image lane's, which is older, and the record knows better
        # than that one (found in review, 2026-10-14)
        b = self.dated(("lane-before-image", "qwen38-sglang.service", 300), ("text-lane", "qwen38-flash.service", 200),
                       ("lane-before-video", "qwen38-image.service", 100))
        b.boot("video")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-flash.service\n")
        self.assertEqual(b.run("text_lane_back video"), "qwen38-flash.service")

    def test_a_newer_link_through_a_newer_one_is_followed(self):
        # the same, with v1.29 back on the 27B and then images after the record: both links newer
        b = self.dated(("text-lane", "qwen38-flash.service", 300), ("lane-before-image", "qwen38-sglang.service", 200),
                       ("lane-before-video", "qwen38-image.service", 100))
        b.boot("video")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-sglang.service\n")

    @staticmethod
    def dated(*files):
        """A box whose config holds these (name, value, age in seconds)."""
        b = Box()
        now = time.time()
        for name, value, age in files:
            (b.cfg / name).write_text(value + "\n")
            os.utime(b.cfg / name, (now - age, now - age))
        return b

    def test_links_older_than_the_record_are_not(self):
        b = Box().files(text_lane="qwen38-flash.service", lane_before_image="qwen38-sglang.service")
        now = time.time()
        os.utime(b.cfg / "lane-before-image", (now - 100, now - 100))
        os.utime(b.cfg / "text-lane", (now - 50, now - 50))
        b.boot("image")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-flash.service\n")

    def test_newer_links_that_lead_nowhere_keep_the_record(self):
        b = Box().files(text_lane="qwen38-flash.service", lane_before_image="qwen38-video.service")
        now = time.time()
        os.utime(b.cfg / "text-lane", (now - 100, now - 100))
        os.utime(b.cfg / "lane-before-image", (now - 50, now - 50))
        b.boot("image")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-flash.service\n")

    def test_a_text_lane_at_boot_is_the_text_lane(self):
        b = Box().files(lane_before_image="qwen38-flash.service")
        b.boot("27b")
        b.run("text_lane_migrate")
        self.assertEqual((b.cfg / "text-lane").read_text(), "qwen38-sglang.service\n")

    def test_the_boot_goes_back_to_an_installed_text_lane(self):
        b = Box(units=("27b", "image")).files(text_lane="qwen38-flash.service")
        self.assertEqual(b.run("text_lane_back image"), "qwen38-sglang.service", "the flash is gone")
        self.assertEqual(Box(units=("flash",)).run("text_lane_back"), "qwen38-flash.service")
        self.assertEqual(Box(units=("image",)).run("text_lane_back image"), "")

    def test_the_boot_goes_back_only_to_a_text_lane_that_can_start(self):
        """Its image and its checkpoint on the box: a lane installed on demand and never loaded,
        or left out beside a side lane at boot (its units, none of its downloads: issue #57), got
        the boot, and pulled its image at boot or started again every 15 s without its checkpoint
        (found in review, 2026-10-21)."""
        noted = Box(units=("27b", "flash", "image"), unready=("flash",)).files(text_lane="qwen38-flash.service")
        self.assertEqual(noted.run("text_lane_back image"), "qwen38-sglang.service", "the flash cannot start")
        self.assertEqual(Box(units=("27b", "flash"), unready=("27b",)).run("text_lane_back"), "qwen38-flash.service")
        self.assertEqual(Box(units=("27b", "flash"), unready=("27b", "flash")).run("text_lane_back"), "")
        # left out (its marker) and able to start, the lane noted keeps the boot: nothing else is wrong with it
        b = Box(units=("27b", "flash", "image")).files(text_lane="qwen38-flash.service")
        (b.cfg / "flash.off").write_text("left out with ./install.sh --no-flash\n")
        self.assertEqual(b.run("text_lane_back image"), "qwen38-flash.service")
        # each half of "can start": the image, then the checkpoint
        b = Box(units=("27b", "flash"), unready=("27b",)).files(text_lane="qwen38-flash.service")
        (b.bin / "docker").write_text("#!/bin/bash\nexit 1\n")
        self.assertEqual(b.run("text_lane_back image"), "", "no image on the box")
        b = Box(units=("27b", "flash"), unready=("27b",)).files(text_lane="qwen38-flash.service")
        for f in (b.d / "hf").rglob("model.safetensors"):
            f.unlink()
        self.assertEqual(b.run("text_lane_back image"), "", "no checkpoint in its cache")

    def test_what_the_uninstall_says_of_the_boot_it_gives_back(self):
        """'as it was before' only of the lane it noted; with none that can start, nothing is
        enabled, and the way to get one is a Load, not `systemctl enable --now qwen38-sglang`."""
        b = Box(units=("27b", "flash", "image")).files(text_lane="qwen38-flash.service")
        out = b.run('text_lane_give_back image "the image lane"')
        self.assertEqual(out, "qwen38-flash.service is enabled at boot again, as it was before the image lane; "
                              "start it now with: sudo systemctl start qwen38-flash.service   (or the cockpit)")
        self.assertEqual((b.d / "sudo").read_text(), "sudo systemctl enable qwen38-flash.service\n")
        b = Box(units=("27b", "flash", "image"), unready=("flash",)).files(text_lane="qwen38-flash.service")
        out = b.run('text_lane_give_back image "the image lane"')
        self.assertIn("qwen38-sglang.service is enabled at boot, a text lane that can start (the one before the "
                      "image lane, qwen38-flash.service, cannot:", out)
        self.assertNotIn("again", out)
        b = Box(units=("27b", "flash", "image"), unready=("27b", "flash"))
        out = b.run('text_lane_give_back image "the image lane"')
        self.assertIn("NOTE: no engine is enabled at boot any more: no text lane on this box can start", out)
        self.assertIn("Load one from the cockpit's Lanes view, or: ./prepare-lane.sh 27b stock, then ./switch-model.sh stock", out)
        self.assertNotIn("enable --now", out)
        self.assertFalse((b.d / "sudo").exists(), "nothing enabled")
        # the way it names is a lane the box has: prepare-lane.sh refuses a 27B the box does not have
        out = Box(units=("flash", "image"), unready=("flash",)).run('text_lane_give_back image "the image lane"')
        self.assertIn("or: ./prepare-lane.sh flash flash, then ./switch-model.sh flash", out)
        out = Box(units=("image",)).run('text_lane_give_back image "the image lane"')
        self.assertIn("or: ./install.sh installs one", out)

    def test_the_lane_the_box_comes_back_to_ready_or_not(self):
        """What opencode's default follows when no text lane can start (vllm-lane.sh): the lane noted, else one
        installed, as text_lane_back said before v1.37.7; the 27B on a box without one was a default to nothing."""
        b = Box(units=("flash", "image"), unready=("flash",)).files(text_lane="qwen38-flash.service")
        self.assertEqual(b.run("text_lane_back image"), "")
        self.assertEqual(b.run("text_lane_home image"), "qwen38-flash.service")
        self.assertEqual(Box(units=("27b", "flash"), unready=("27b", "flash")).run("text_lane_home"), "qwen38-sglang.service")
        self.assertEqual(Box(units=("image",)).run("text_lane_home image"), "")

    def test_a_lane_whose_unit_is_gone_gets_no_boot_however_ready_its_files(self):
        b = Box(units=("27b", "image"), unready=("27b",)).files(text_lane="qwen38-flash.service")
        fixture.put_ready(b.units, b.cfg, b.d / "hf", ["flash"])        # its launcher and checkpoint stay
        (b.units / "qwen38-flash.service").unlink()                      # its unit went
        self.assertEqual(b.run("text_lane_back image"), "")

    def test_a_lane_serving_older_files_is_told(self):
        b = Box()
        unit, runtime = b.units / "qwen38-image.service", b.cfg / "runtime-changed"
        runtime.write_text("")
        started = int(time.time()) - 100
        (b.bin / "systemctl").write_text(
            "#!/bin/bash\n"
            'case "$*" in *ActiveState*) echo "${STATE:-active}";; '
            f'*ExecMainStartTimestamp*) echo "@{started}";; esac\n')
        older = f'lane_serves_older qwen38-image.service "{unit}" "{runtime}" "{b.cfg}/absent" && echo OLDER || echo SAME'
        for f in (unit, runtime):
            os.utime(f, (started - 50, started - 50))
        self.assertEqual(b.run(older), "SAME")
        for f in (unit, runtime):                        # either one newer than the start is enough
            with self.subTest(newer=f.name):
                os.utime(f, (started + 50, started + 50))
                self.assertEqual(b.run(older), "OLDER")
                self.assertEqual(b.run("STATE=inactive " + older), "SAME", "a lane that does not serve serves nothing older")
                os.utime(f, (started - 50, started - 50))

    def test_stale_since_is_true_for_a_service_that_does_not_serve(self):
        # install.sh's own (the proxy's), moved here: it restarts a service that is down, or
        # that serves files older than these
        b = Box()
        unit = b.units / "qwen38-keepalive.service"
        unit.write_text("[Service]\n")
        started = int(time.time()) - 100
        (b.bin / "systemctl").write_text(
            "#!/bin/bash\n"
            'case "$*" in *ActiveState*) echo "${STATE:-active}";; '
            f'*ExecMainStartTimestamp*) echo "${{STARTED-@{started}}}";; esac\n')
        stale = f'stale_since qwen38-keepalive.service "{unit}" && echo STALE || echo FRESH'
        os.utime(unit, (started - 50, started - 50))
        self.assertEqual(b.run(stale), "FRESH")
        self.assertEqual(b.run("STATE=inactive " + stale), "STALE")
        self.assertEqual(b.run("STARTED= " + stale), "STALE", "no start time read: restarted, as before")
        os.utime(unit, (started + 50, started + 50))
        self.assertEqual(b.run(stale), "STALE")


class EverySequenceOfSwitches(unittest.TestCase):
    """Every sequence of lanes up to six long (9,330) that starts on a text lane (the box's first
    install does), switched through switch-model.sh's own notes: install.sh resolves the last
    text lane served, and the uninstall of the side lane at boot gives the boot back to it."""

    BOX = None

    def replay(self, seq):
        """One shell per sequence: each switch's own lines, then the box boots the next lane."""
        b = self.BOX = self.BOX or Box()
        (b.cfg / "text-lane").unlink(missing_ok=True)   # a fresh box for each sequence
        boot = lambda lane: f'printf "%s\\n" {UNIT[lane]} > "{b.d}/enabled"\n'
        steps = [boot(seq[0]), f'text_lane_note "{UNIT[seq[0]]}"\n']   # the first install notes its lane
        for nxt in seq[1:]:
            if nxt in SIDES:
                steps.append(NOTES[nxt])            # noted while the lane it leaves still boots
            else:
                steps.append(f'TARGET_UNIT_NAME="{UNIT[nxt]}"\ntext_lane_note "$TARGET_UNIT_NAME"\n')
            steps.append(boot(nxt))
        steps.append(RESOLVE + 'echo "R=$TEXT_LANE_LAST"\n')
        if seq[-1] in SIDES:
            steps.append(f'echo "B=$(text_lane_back {seq[-1]})"\n')
        out = dict(ln.split("=", 1) for ln in b.run("".join(steps)).splitlines() if "=" in ln)
        return out["R"], out.get("B")

    def test_every_sequence(self):
        n, bad = 0, []
        for length in range(2, 7):
            for tail in itertools.product(("27b", "flash", "image", "video", "kolibri", "nemotron", "nemotron-super"),
                                          repeat=length - 1):
                seq = (("flash", "27b")[length % 2],) + tail
                if any(a == b for a, b in zip(seq, seq[1:])):
                    continue
                truth = UNIT[[lane for lane in seq if lane not in SIDES][-1]]
                n += 1
                resolved, back = self.replay(seq)
                if resolved != truth or back not in (None, truth):
                    bad.append((seq, resolved, back, truth))
        self.assertGreater(n, 1000)
        self.assertEqual(bad[:5], [], f"{len(bad)} of {n} sequences")


if __name__ == "__main__":
    unittest.main()
