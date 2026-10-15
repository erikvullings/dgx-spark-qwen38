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


# The image and video lanes' installers have the same two halves: one test each, with a subTest
# per lane (the CI counts every test once against what unittest runs, and a test class inheriting
# another's tests ran them twice under one name).
LANES_IN_TWO_HALVES = (("install-image.sh", ".local/share/qwen38-image", "qwen38-image.service"),
                       ("install-video.sh", ".local/share/qwen38-video", "qwen38-video.service"))


class TheImageAndVideoLanesInTwoHalves(unittest.TestCase):
    def test_units_only_puts_in_the_unit_and_nothing_heavy(self):
        for installer, lane, unit in LANES_IN_TWO_HALVES:
            with self.subTest(installer=installer):
                box = Box(installer, root_ok=True)
                rc, out, calls = box.run("--units-only")
                self.assertEqual(rc, 0, out)
                self.assertTrue((box.units / unit).exists(), out)
                self.assertFalse((box.home / lane / "venv").exists(), "a venv was made")
                self.assertFalse([c for c in calls if c.startswith(("git ", "python -m pip", "python -m venv"))], calls)
                self.assertFalse([c for c in calls if "ckpt-fetch.py" in c], calls)
                self.assertIn("first Load", out)

    def test_prepare_builds_the_runtime_with_no_sudo_and_no_unit(self):
        for installer, lane, unit in LANES_IN_TWO_HALVES:
            with self.subTest(installer=installer):
                box = Box(installer)
                rc, out, calls = box.run("--prepare")
                self.assertEqual(rc, 0, out)
                self.assertTrue((box.home / lane / "venv/bin/python").exists(), out)
                self.assertTrue((box.home / lane / "sglang-wheel").exists(), out)
                self.assertFalse([c for c in calls if c.startswith(("sudo ", "systemctl ", "docker "))], calls)
                self.assertFalse((box.units / unit).exists(), "--prepare wrote the unit")
                self.assertFalse([c for c in calls if "ckpt-fetch.py" in c], calls)

    def test_the_two_halves_are_asked_for_one_at_a_time(self):
        for installer, _lane, _unit in LANES_IN_TWO_HALVES:
            with self.subTest(installer=installer):
                rc, out, calls = Box(installer).run("--units-only", "--prepare")
                self.assertNotEqual(rc, 0)
                self.assertEqual(calls, [])


class TheVideoLaneTurboAdapter(unittest.TestCase):
    def test_prepare_fetches_the_turbo_adapter_and_not_the_checkpoint(self):
        # the switch fetches the checkpoint; the adapter is the installer's alone to fetch
        rc, out, calls = Box("install-video.sh").run("--prepare")
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

    def run_it(self, *args, units=("image", "video", "kolibri")):
        d = pathlib.Path(tempfile.mkdtemp(prefix="prepare-lane-"))
        (d / "units").mkdir()
        for lane in units:
            (d / "units" / f"qwen38-{lane}.service").write_text("[Unit]\n")
        (d / "prepare-lane.sh").write_text((REPO / "prepare-lane.sh").read_text().replace("/etc/systemd/system/", f"{d}/units/"))
        for lane in ("image", "video", "kolibri"):
            (d / f"install-{lane}.sh").write_text(f'echo "install-{lane}.sh $*"\n')
        r = subprocess.run(["bash", str(d / "prepare-lane.sh"), *args], capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr).strip()

    def test_each_lane_runs_its_installers_prepare(self):
        for lane in ("image", "video", "kolibri"):
            with self.subTest(lane=lane):
                self.assertEqual(self.run_it(lane), (0, f"install-{lane}.sh --prepare"))

    def test_a_lane_with_no_unit_is_not_prepared(self):
        # its units come with ./install.sh; a preparation alone would build what nothing runs
        for lane in ("image", "video", "kolibri"):
            with self.subTest(lane=lane):
                rc, out = self.run_it(lane, units=())
                self.assertEqual(rc, 1, out)
                self.assertIn("not installed on this box", out)
                self.assertNotIn("--prepare", out)

    def test_anything_else_is_refused(self):
        for args in ((), ("27B",), ("../install",), ("image", "--uninstall"), ("image", "image"), ("",),
                     ("flash", "flash-nvda", "x")):
            with self.subTest(args=args):
                rc, out = self.run_it(*args)
                self.assertEqual(rc, 2, out)
                self.assertIn("usage", out)



class TextBox:
    """prepare-lane.sh <flash|27b>, run as written in a HOME of the test's own, against a docker
    that records its calls and knows which images are on the box (one is, once pulled), a df that
    gives the room the test says, and a switch-model.sh and a patch-yarn.py that record theirs.
    The launcher and the unit are written as their templates write them."""

    PINS = dict(re.findall(r'^((?:FLASH(?:_NVDA|_UNC)?|STOCK|UNC|FP8|UNCFP8)_(?:REPO|REV))="([^"]+)"',
                           (REPO / "install.sh").read_text(), re.M))

    def __init__(self, image_here=False, free_gb=1000):
        self.t = t = pathlib.Path(tempfile.mkdtemp(prefix="prepare-text-"))
        self.d = d = t / "repo"
        d.mkdir()
        for f in ("lanes.sh", "install.sh"):
            (d / f).write_text((REPO / f).read_text())
        self.units = t / "units"
        self.units.mkdir()
        (d / "prepare-lane.sh").write_text((REPO / "prepare-lane.sh").read_text().replace("/etc/systemd/system/", f"{self.units}/"))
        (d / "switch-model.sh").write_text(f'echo "switch-model.sh $*" >> "{t}/calls"\n')
        (d / "patch-yarn.py").write_text(f'import sys\nopen("{t}/calls", "a").write("patch-yarn.py " + " ".join(sys.argv[1:]) + "\\n")\n')
        self.hf, self.ple = t / "hf", t / "ple"
        self.hf.mkdir()
        self.ple.mkdir()
        self.cfg = t / "home" / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (t / "docker-root").mkdir()
        (t / "bin").mkdir()
        self.stub("docker", f'echo "docker $*" >> "{t}/calls"\n'
                  'case "$1 $2" in\n'
                  f'  "info "*) echo "{t}/docker-root" ;;\n'
                  '  "image inspect") case "$3" in qwen38-pinned:*) exit 1 ;; esac\n'
                  f'    [ -e "{t}/pulled" ] || [ {int(image_here)} = 1 ] || exit 1; echo sha256:abc ;;\n'
                  f'  "pull "*) touch "{t}/pulled" ;;\n'
                  'esac\nexit 0\n')
        self.stub("df", f'printf "Avail\\n{free_gb}G\\n"')

    def stub(self, name, body):
        (self.t / "bin" / name).write_text("#!/bin/sh\n" + body)
        (self.t / "bin" / name).chmod(0o755)

    def flash(self, model, image, map_line=True):
        (self.cfg / "launch-flash.sh").write_text(
            "exec /usr/bin/docker run --rm --name qwen38-flash --gpus all \\\n"
            f"  -v {self.hf}:/root/.cache/huggingface \\\n  -v {self.cfg}:/out \\\n  -v {self.ple}:/ple \\\n"
            f"  -e HF_HUB_OFFLINE=1 \\\n  {image} \\\n  python3 -m sglang.launch_server \\\n"
            f"    --model-path {model} --tp-size 1 \\\n"
            + (f'[ -s "{self.cfg}/token-map-65536.pt" ] && TIER+=(--speculative-token-map /out/token-map-65536.pt)\n' if map_line else ""))

    def unit27(self, model, image, window="1m", draft="org/Qwen3.8-27B-DFlash2", drev="d" * 40):
        (self.units / "qwen38-sglang.service").write_text(
            "[Service]\nExecStart=/usr/bin/docker run --rm --name qwen38-sglang --gpus all \\\n"
            f"  -v {self.cfg}/sglang-cache:/cache \\\n  -v {self.hf}:/root/.cache/huggingface \\\n  -v {self.cfg}:/out \\\n"
            f"  {image} \\\n  python3 -m sglang.launch_server \\\n"
            f"    --trust-remote-code --model-path {model} --tp-size 1 \\\n"
            + ("    --context-length 1010000 \\\n" if window == "1m" else "")
            + f"    --speculative-algorithm DFLASH --speculative-draft-model-path {draft} \\\n"
            f"    --speculative-draft-model-revision {drev} \\\n")

    def run(self, lane, *rest, **env):
        r = subprocess.run(["bash", str(self.d / "prepare-lane.sh"), lane, *rest], capture_output=True, text=True, timeout=60,
                           env={"PATH": f"{self.t}/bin:/usr/bin:/bin", "HOME": str(self.t / "home"), **env})
        calls = (self.t / "calls").read_text().splitlines() if (self.t / "calls").exists() else []
        return r.returncode, r.stdout + r.stderr, calls


def heads(calls):
    return [" ".join(c.split()[0:2]) for c in calls]


class TheFlashPreparation(unittest.TestCase):
    """prepare-lane.sh flash: the image the launcher runs pulled (its room checked first) and
    tagged, the checkpoint of the target it serves fetched by the switch's own --fetch, the room
    of the PLE table it writes at every boot checked after it, and the reduced draft vocabulary
    built from it when the launcher takes one."""

    IMG = "lmsysorg/sglang@sha256:" + "c" * 64

    def box(self, repo_var="FLASH", map_line=True, image="", **kw):
        b = TextBox(**kw)
        model = b.PINS[f"{repo_var}_REPO"]
        (b.hf / "hub" / ("models--" + model.replace("/", "--")) / "snapshots" / b.PINS[f"{repo_var}_REV"]).mkdir(parents=True)
        b.flash(model, image or self.IMG, map_line)
        return b

    def test_image_checkpoint_and_vocabulary_in_that_order(self):
        rc, out, calls = self.box().run("flash")
        self.assertEqual(rc, 0, out)
        self.assertEqual(heads(calls), ["docker image", "docker info", "docker pull", "docker image", "docker image",
                                        "docker tag", "switch-model.sh flash", "docker run"])
        self.assertIn(f"docker pull {self.IMG}", calls)
        # tagged as install.sh tags what it pulls, so a docker image prune leaves it alone
        self.assertIn(f"docker tag {self.IMG} qwen38-pinned:flash-cccccccccccc", calls)
        self.assertIn("switch-model.sh flash --fetch", calls)
        self.assertIn(f"/snapshots/{TextBox.PINS['FLASH_REV']} --out /out/token-map-65536.pt --size 65536", calls[-1])

    def test_the_target_follows_the_model_the_launcher_serves(self):
        rc, out, calls = self.box(repo_var="FLASH_NVDA", image_here=True).run("flash")
        self.assertEqual(rc, 0, out)
        self.assertIn("switch-model.sh flash-nvda --fetch", calls)
        self.assertNotIn("docker pull", heads(calls))

    def test_the_target_the_load_loads_is_the_one_fetched(self):
        # the Load names its target: that checkpoint is fetched and the vocabulary built from it,
        # not the launcher's (found in review, 2026-10-15: a first Load of flash-nvda fetched the
        # default, 135 GB, then its switch fetched the NVIDIA export)
        b = self.box()                                                  # the launcher serves the default
        nv, nv_rev = b.PINS["FLASH_NVDA_REPO"], b.PINS["FLASH_NVDA_REV"]
        (b.hf / "hub" / ("models--" + nv.replace("/", "--")) / "snapshots" / nv_rev).mkdir(parents=True)
        rc, out, calls = b.run("flash", "flash-nvda")
        self.assertEqual(rc, 0, out)
        self.assertIn("switch-model.sh flash-nvda --fetch", calls)
        self.assertNotIn("switch-model.sh flash --fetch", calls)
        self.assertIn(f"models--{nv.replace('/', '--')}/snapshots/{nv_rev} --out /out/token-map-65536.pt", calls[-1])

    def test_a_target_of_another_lane_is_refused_before_anything(self):
        for target in ("stock", "image", "flash-turbo"):
            with self.subTest(target=target):
                rc, out, calls = self.box().run("flash", target)
                self.assertEqual(rc, 1, out)
                self.assertIn("is not a flash lane target", out)
                self.assertEqual(calls, [])

    def test_an_image_an_operator_named_by_tag_is_read_too(self):
        tag = "lmsysorg/sglang:dev-qwen38-next-local"
        rc, out, calls = self.box(image=tag).run("flash")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"docker pull {tag}", calls)
        self.assertNotIn("docker tag", heads(calls))          # a tag needs none

    def test_no_vocabulary_asked_none_built(self):
        rc, out, calls = self.box(map_line=False).run("flash")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("docker run", heads(calls))

    def test_no_room_for_the_image_is_refused_before_its_pull(self):
        rc, out, calls = self.box(free_gb=20).run("flash")
        self.assertEqual(rc, 1, out)
        self.assertIn("need ~35 GB free", out)
        self.assertEqual(heads(calls), ["docker image", "docker info"])

    def test_no_room_for_the_ple_table_is_refused_after_the_checkpoint(self):
        rc, out, calls = self.box(image_here=True, free_gb=20).run("flash")
        self.assertEqual(rc, 1, out)
        self.assertIn("for the flash lane's PLE table", out)
        self.assertEqual(heads(calls)[-1], "switch-model.sh flash")   # the checkpoint came; no vocabulary built
        b = self.box(image_here=True, free_gb=20)
        (b.ple / "ple_table_0.bin").write_text("t")                   # a table already there takes its own room
        self.assertEqual(b.run("flash")[0], 0)


class TheTwentySevenBPreparation(unittest.TestCase):
    """prepare-lane.sh 27b, the 27B beside a serving flash: the image its unit runs pulled and
    tagged, its DFlash2 draft fetched in that image, its config patched for the 1M window the unit
    serves (restored for the native one), and the checkpoint of the target it serves fetched by
    the switch's own --fetch, which patches the target's config at the Load."""

    IMG = "lmsysorg/sglang@sha256:" + "e" * 64
    DRAFT, DREV = "org/Qwen3.8-27B-DFlash2", "d" * 40

    def box(self, repo_var="UNC", window="1m", **kw):
        b = TextBox(**kw)
        b.unit27(b.PINS[f"{repo_var}_REPO"], self.IMG, window, self.DRAFT, self.DREV)
        return b

    def test_image_draft_its_config_and_checkpoint_in_that_order(self):
        b = self.box()
        rc, out, calls = b.run("27b")
        self.assertEqual(rc, 0, out)
        self.assertEqual(heads(calls), ["docker image", "docker info", "docker pull", "docker image", "docker image",
                                        "docker tag", "docker run", "patch-yarn.py " + str(b.hf), "switch-model.sh uncensored"])
        run = next(c for c in calls if c.startswith("docker run"))
        self.assertIn(f"-v {b.hf}:/hf {self.IMG} /opt/qwen38/ckpt-fetch.py {self.DRAFT} {self.DREV} --refs-main fill", run)
        self.assertIn(f"patch-yarn.py {b.hf} {self.DRAFT} {self.DREV}", calls)
        self.assertIn("switch-model.sh uncensored --fetch", calls)
        self.assertIn(f"docker tag {self.IMG} qwen38-pinned:27b-eeeeeeeeeeee", calls)

    def test_a_native_unit_puts_the_drafts_config_back(self):
        b = self.box(window="native", image_here=True)
        rc, out, calls = b.run("27b")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"patch-yarn.py --restore {b.hf} {self.DRAFT} {self.DREV}", calls)

    def test_the_target_follows_the_model_the_unit_serves(self):
        for repo_var, target in (("STOCK", "stock"), ("FP8", "fp8"), ("UNCFP8", "uncensored-fp8")):
            with self.subTest(target=target):
                rc, out, calls = self.box(repo_var=repo_var, image_here=True).run("27b")
                self.assertEqual(rc, 0, out)
                self.assertEqual(calls[-1], f"switch-model.sh {target} --fetch")

    def test_the_target_the_load_loads_is_the_one_fetched(self):
        # an uncensored unit, a Load of fp8: fp8 is fetched, not the unit's (2026-10-15)
        rc, out, calls = self.box(repo_var="UNC", image_here=True).run("27b", "fp8")
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls[-1], "switch-model.sh fp8 --fetch")
        rc, out, calls = self.box().run("27b", "flash")
        self.assertEqual(rc, 1, out)
        self.assertIn("is not a 27B lane target", out)
        self.assertEqual(calls, [])


    def test_a_token_goes_by_its_name_only(self):
        rc, out, calls = self.box(image_here=True).run("27b", HF_TOKEN="hf_value_never_shown")
        self.assertEqual(rc, 0, out)
        run = next(c for c in calls if c.startswith("docker run"))
        self.assertIn(" -e HF_TOKEN ", run)
        self.assertNotIn("hf_value_never_shown", "\n".join(calls) + out)

    def test_what_it_cannot_prepare_is_refused(self):
        b = TextBox()
        b.unit27("org/a-custom-27b", self.IMG)
        rc, out, _ = b.run("27b")
        self.assertEqual(rc, 1, out)
        self.assertIn("which no 27B target names", out)
        rc, out, _ = TextBox().run("27b")                                   # no unit at all
        self.assertEqual(rc, 1, out)
        self.assertIn("the 27B lane is not installed", out)


if __name__ == "__main__":
    unittest.main()
