#!/usr/bin/env python3
"""A lane installed on demand comes in two halves (decided 2026-10-14): --units-only puts in
what needs root (the unit, its start scripts) and nothing heavy, and --prepare, which the
cockpit's first Load of the lane runs as the user, builds its runtime with no sudo at all.
The weights come with that Load's switch, as for any Load.

The installers run whole here, on a copy whose unit path is the test's own, with fakes for
everything that acts on the machine: sudo, systemctl and docker record their calls (sudo
runs what follows it only when the test allows root), python3 and git stand in for the venv,
the wheel and the source, and uname says aarch64 on any machine the suite runs on.
"""
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]

# python3 and the venv's python: the checks the installers ask answer yes, a venv is a folder
# with this same script as its python, pip records what it would install, an import of sglang
# fails until the wheel is "installed", and the heredoc checks of the runtime pass.
FAKE_PY = r'''#!/bin/bash
printf 'python %s\n' "$*" >> "$CALLS"
case "$1" in
  -m)
    if [ "$2" = venv ]; then d="${@: -1}"; mkdir -p "$d/bin"; cp "$0" "$d/bin/python"; ln -sf python "$d/bin/pip"; exit 0; fi
    if [ "$2" = pip ]; then [ "$3" = install ] && case "$*" in *sglang*) touch "$FAKE_STATE/sglang" ;; esac; exit 0; fi ;;
  -c)
    case "$2" in
      *sysconfig*) echo "$FAKE_STATE/include"; exit 0 ;;
      *"import sglang, pathlib"*) exit 0 ;;
      *"import sglang"*) [ -e "$FAKE_STATE/sglang" ]; exit ;;
    esac
    exit 0 ;;
  -) cat > /dev/null; exit 0 ;;
esac
case "$(basename "$0")" in pip) case "$*" in *sglang*) touch "$FAKE_STATE/sglang" ;; esac ;; esac
exit 0
'''
FAKE_GIT = r'''#!/bin/bash
printf 'git %s\n' "$*" >> "$CALLS"
if [ "$1" = clone ]; then mkdir -p "${@: -1}/.git"; exit 0; fi
case "$*" in *rev-parse*) echo "$FAKE_PIN"; exit 0 ;; *"apply --reverse --check"*) exit 0 ;; esac
exit 0
'''
RECORD = '#!/bin/bash\nprintf "%s %s\\n" "$(basename "$0")" "$*" >> "$CALLS"\n'
FAKE_SUDO = RECORD + '[ -n "$ROOT_OK" ] && exec "$@"\nexit 1\n'


class Box:
    def __init__(self, installer, root_ok=False):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="on-demand-"))
        self.home = self.t / "home"
        (self.home / ".config/qwen38").mkdir(parents=True)
        (self.home / ".config/qwen38/api-key").write_text("k\n")
        self.state = self.t / "state"
        (self.state / "include").mkdir(parents=True)
        (self.state / "include" / "Python.h").write_text("")
        self.units = self.t / "units"
        self.units.mkdir()
        repo = self.t / "repo"
        repo.mkdir()
        text = (REPO / installer).read_text()
        old = '"/etc/systemd/system/$UNIT"'
        assert old in text, f"{installer} no longer names its unit that way"
        (repo / installer).write_text(text.replace(old, f'"{self.units}/$UNIT"'))
        for f in REPO.iterdir():
            if f.is_file() and f.name != installer and (f.suffix in (".sh", ".py", ".template") or f.name.endswith(".service.template")):
                (repo / f.name).write_text(f.read_text())
        for d in ("image-sglang", "video-sglang"):
            if (REPO / d).is_dir():
                (repo / d).mkdir()
                for f in (REPO / d).iterdir():
                    (repo / d / f.name).write_text(f.read_text())
        self.script = repo / installer
        self.bin = self.t / "bin"
        self.bin.mkdir()
        for name, body in (("python3", FAKE_PY), ("git", FAKE_GIT), ("sudo", FAKE_SUDO),
                           ("systemctl", RECORD + '[ "$1" = daemon-reload ] && exit 0\nexit 1\n'),
                           ("docker", RECORD + "exit 1\n"),
                           ("nvidia-smi", RECORD + "exit 0\n"), ("uname", "#!/bin/sh\necho aarch64\n"),
                           ("ffmpeg", RECORD + "exit 0\n"), ("ffprobe", RECORD + "exit 0\n")):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        self.calls = self.t / "calls"
        self.root_ok = root_ok
        pin = re.search(r'SGLANG_DIFFUSION_PIN:-([0-9a-f]{40})', text)
        self.pin = pin.group(1) if pin else ""

    def run(self, *args):
        env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home), "CALLS": str(self.calls),
               "FAKE_STATE": str(self.state), "FAKE_PIN": self.pin, "HF_CACHE": str(self.t / "hf")}
        if self.root_ok:
            env["ROOT_OK"] = "1"
        r = subprocess.run(["bash", str(self.script), *args], capture_output=True, text=True, env=env, timeout=120)
        calls = self.calls.read_text().splitlines() if self.calls.exists() else []
        return r.returncode, r.stdout + r.stderr, calls


class TheImageLaneInTwoHalves(unittest.TestCase):
    INSTALLER = "install-image.sh"
    LANE = ".local/share/qwen38-image"
    UNIT = "qwen38-image.service"

    def test_units_only_puts_in_the_unit_and_nothing_heavy(self):
        box = Box(self.INSTALLER, root_ok=True)
        rc, out, calls = box.run("--units-only")
        self.assertEqual(rc, 0, out)
        self.assertTrue((box.units / self.UNIT).exists(), out)
        self.assertFalse((box.home / self.LANE / "venv").exists(), "a venv was made")
        self.assertFalse([c for c in calls if c.startswith(("git ", "python -m pip", "python -m venv"))], calls)
        self.assertFalse([c for c in calls if "ckpt-fetch.py" in c], calls)
        self.assertIn("first Load", out)

    def test_prepare_builds_the_runtime_with_no_sudo_and_no_unit(self):
        box = Box(self.INSTALLER)
        rc, out, calls = box.run("--prepare")
        self.assertEqual(rc, 0, out)
        self.assertTrue((box.home / self.LANE / "venv/bin/python").exists(), out)
        self.assertTrue((box.home / self.LANE / "sglang-wheel").exists(), out)
        self.assertFalse([c for c in calls if c.startswith(("sudo ", "systemctl ", "docker "))], calls)
        self.assertFalse((box.units / self.UNIT).exists(), "--prepare wrote the unit")
        self.assertFalse([c for c in calls if "ckpt-fetch.py" in c], calls)

    def test_the_two_halves_are_asked_for_one_at_a_time(self):
        rc, out, calls = Box(self.INSTALLER).run("--units-only", "--prepare")
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])



class TheVideoLaneInTwoHalves(TheImageLaneInTwoHalves):
    INSTALLER = "install-video.sh"
    LANE = ".local/share/qwen38-video"
    UNIT = "qwen38-video.service"

    def test_prepare_fetches_the_turbo_adapter_and_not_the_checkpoint(self):
        # the switch fetches the checkpoint; the adapter is the installer's alone to fetch
        rc, out, calls = Box(self.INSTALLER).run("--prepare")
        self.assertEqual(rc, 0, out)
        self.assertIn("python - larryvrh/MiniMax-H3-Turbo-Lora", "\n".join(calls))



class TheKolibriLaneInTwoHalves(unittest.TestCase):
    """Kolibri's runtime is an image built on the box: --prepare builds it as the user (docker,
    no sudo). Here the image is already built, as the fake docker says; the build itself is
    the step's own lines, which call no sudo."""

    def box(self, **kw):
        box = Box("install-kolibri.sh", **kw)
        (box.bin / "docker").write_text(RECORD + 'case "$1" in info|images) exit 0 ;; image) exit 0 ;; esac\nexit 1\n')
        for f in ("engine-secrets.sh", "vllm_engine_auth.py", "qwen38-kolibri.service.template"):
            assert (box.script.parent / f).exists(), f
        return box

    def test_units_only_puts_in_the_unit_and_builds_nothing(self):
        box = self.box(root_ok=True)
        rc, out, calls = box.run("--units-only")
        self.assertEqual(rc, 0, out)
        self.assertTrue((box.units / "qwen38-kolibri.service").exists(), out)
        self.assertFalse([c for c in calls if c.startswith(("docker build", "docker pull", "docker run"))], calls)
        self.assertIn("first Load", out)

    def test_prepare_readies_the_image_with_no_sudo_and_no_unit(self):
        box = self.box()
        rc, out, calls = box.run("--prepare")
        self.assertEqual(rc, 0, out)
        self.assertFalse([c for c in calls if c.startswith(("sudo ", "systemctl ", "docker run"))], calls)
        self.assertFalse((box.units / "qwen38-kolibri.service").exists())
        self.assertIn("image is ready", out)

    def test_the_build_step_calls_no_sudo(self):
        text = (REPO / "install-kolibri.sh").read_text()
        build = text[text.index('step "2/5 Image ($LOCAL_IMAGE)"'):text.index('step "3/5 Checkpoint')]
        self.assertNotRegex(build, r"(^|[;&|(]\s*)sudo\b")



class TheDispatcher(unittest.TestCase):
    """prepare-lane.sh, what the cockpit's prepare action runs: the lane is an argument of its
    own (the cockpit never builds one argument out of a parameter), mapped to its installer's
    --prepare, and nothing else is taken."""

    def run_it(self, *args):
        d = pathlib.Path(tempfile.mkdtemp(prefix="prepare-lane-"))
        (d / "prepare-lane.sh").write_text((REPO / "prepare-lane.sh").read_text())
        for lane in ("image", "video", "kolibri"):
            (d / f"install-{lane}.sh").write_text(f'echo "install-{lane}.sh $*"\n')
        r = subprocess.run(["bash", str(d / "prepare-lane.sh"), *args], capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr).strip()

    def test_each_lane_runs_its_installers_prepare(self):
        for lane in ("image", "video", "kolibri"):
            with self.subTest(lane=lane):
                self.assertEqual(self.run_it(lane), (0, f"install-{lane}.sh --prepare"))

    def test_anything_else_is_refused(self):
        for args in ((), ("flash",), ("../install",), ("image", "--uninstall"), ("",)):
            with self.subTest(args=args):
                rc, out = self.run_it(*args)
                self.assertEqual(rc, 2, out)
                self.assertIn("usage", out)


if __name__ == "__main__":
    unittest.main()
