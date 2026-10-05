#!/usr/bin/env python3
"""install-speech.sh: its pins, its core split, its renders and its guards, run as written.

The speech lane runs on the CPU alone, on the small cores, beside whichever lane serves
(docs/speech-lane.md). These read the installer's own lines and run its functions in bash
against fake lscpu output and fake paths: nothing is installed, nothing is downloaded.
"""
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (REPO / "install-speech.sh").read_text()
UNIT_T = (REPO / "qwen38-speech.service.template").read_text()
LAUNCH_T = (REPO / "qwen38-speech-launch.sh.template").read_text()
REQS = (REPO / "speech-requirements.txt").read_text()


def function(name):
    """The text of a bash function of the installer, from `name(){` to its closing brace:
    the one line of a function written on one line, or down to the first `}` alone."""
    start = SCRIPT.index(f"{name}(){{")
    line = SCRIPT[start:SCRIPT.index("\n", start)]
    if line.rstrip().endswith("}"):
        return line + "\n"
    return SCRIPT[start:SCRIPT.index("\n}\n", start) + 3]


def bash(script, stdin=""):
    r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + script], input=stdin,
                       capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout, r.stderr


GB10 = "CPU MAXMHZ\n" + "".join(f"{c} {3900.0 if c in range(5, 10) or c in range(15, 20) else 2808.0:.4f}\n"
                                for c in range(20))


class Pins(unittest.TestCase):
    def test_every_download_is_pinned_and_checked(self):
        self.assertRegex(SCRIPT, r'LLAMA_CPP_PIN="\$\{SPEECH_LLAMA_CPP_PIN:-[0-9a-f]{40}\}"')
        revs = re.findall(r'ASR_REV="([0-9a-f]+)"', SCRIPT)
        shas = re.findall(r'_SHA="([0-9a-f]+)"', SCRIPT)
        self.assertEqual(len(revs), 2)
        self.assertTrue(all(len(r) == 40 for r in revs), revs)
        self.assertEqual(len(shas), 6)                 # two recognizers x (model, mmproj), Kokoro x 2
        self.assertTrue(all(len(s) == 64 for s in shas), shas)
        fetches = re.findall(r'^fetch "([^"]+)" "([^"]+)" "([^"]+)"$', SCRIPT, re.M)
        self.assertEqual(len(fetches), 4)
        for url, _, sha in fetches:
            self.assertTrue(sha.startswith("$") and sha.endswith("_SHA"), sha)
            if "huggingface.co" in url:
                self.assertIn("/resolve/$ASR_REV/", url)

    def test_a_file_that_does_not_match_is_not_kept(self):
        fetch = function("fetch")
        self.assertIn('rm -f "$out.part"; die', fetch)
        self.assertLess(fetch.index("sha256sum \"$out.part\""), fetch.index('mv "$out.part" "$out"'))

    def test_the_python_runtime_is_exact_and_wheels_only(self):
        lines = [ln for ln in REQS.splitlines() if ln and not ln.startswith("#")]
        self.assertTrue(lines)
        for ln in lines:
            self.assertRegex(ln, r"^[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.]*$")
        self.assertIn('pip" install --quiet --only-binary :all: -r "$HERE/speech-requirements.txt"', SCRIPT)


class TheSmallCores(unittest.TestCase):
    def cores(self, lscpu):
        code, out, err = bash(function("small_cores") + "\nsmall_cores | paste -sd' ' -", lscpu.split("\n", 1)[1])
        self.assertEqual(code, 0, err)
        return out.strip()

    def test_the_gb10s_small_cores_are_its_ten_a725(self):
        self.assertEqual(self.cores(GB10), "0 1 2 3 4 10 11 12 13 14")

    def test_one_kind_of_core_is_used_whole(self):
        self.assertEqual(self.cores("CPU MAXMHZ\n" + "".join(f"{c} 3000.0\n" for c in range(8))), "0 1 2 3 4 5 6 7")

    def test_too_few_slow_cores_means_all_of_them(self):
        lscpu = "CPU MAXMHZ\n0 1000.0\n1 1000.0\n" + "".join(f"{c} 3000.0\n" for c in range(2, 8))
        self.assertEqual(self.cores(lscpu), "0 1 2 3 4 5 6 7")

    def test_the_build_takes_no_more_jobs_than_the_free_memory_allows(self):
        start = SCRIPT.index("  JOBS=$((")
        end = SCRIPT.index("\n", SCRIPT.index('[ "$JOBS" -ge 1 ]', start)) + 1
        block = SCRIPT[start:end]
        for avail_gb, want in ((17, 5), (60, 10), (2, 1), (0, 1)):
            with self.subTest(avail_gb=avail_gb), tempfile.NamedTemporaryFile("w") as meminfo:
                meminfo.write(f"MemTotal: 130000000 kB\nMemAvailable: {avail_gb * 1048576 + 1000} kB\n")
                meminfo.flush()
                code, out, err = bash("CORE_LIST=(0 1 2 3 4 10 11 12 13 14)\n" +
                                      block.replace("/proc/meminfo", meminfo.name) + 'echo "$JOBS"')
                self.assertEqual(code, 0, err)
                self.assertEqual(out.strip(), str(want))

    def test_the_split_gives_each_engine_its_own_cores(self):
        start = SCRIPT.index('CORES="$(lscpu')
        end = SCRIPT.index("UNIT_CPUS=", start)
        end = SCRIPT.index("\n", end) + 1
        block = SCRIPT[start:end].replace("lscpu -e=CPU,MAXMHZ 2>/dev/null", "cat")
        code, out, err = bash("die(){ echo \"$*\" >&2; exit 1; }\n" + function("small_cores") + block +
                              'echo "$ASR_CPUS|$TTS_CPUS|$ASR_THREADS|$TTS_THREADS|$UNIT_CPUS"', GB10)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), "0,1,2,3,4|10,11,12,13,14|5|5|0 1 2 3 4 10 11 12 13 14")


RENDER_ENV = """
HOME=/home/u; LANE_DIR=/home/u/.local/share/qwen38-speech; VENV=$LANE_DIR/venv; MODELS=$LANE_DIR/models
PORT=31000; ASR_PORT=31001; BIND=127.0.0.1; VOICE=af_heart; UNIT_CPUS="0 1 2 3 4 10 11 12 13 14"
ASR_CPUS=0,1,2,3,4; TTS_CPUS=10,11,12,13,14; ASR_THREADS=5; TTS_THREADS=5
LLAMA_SERVER=$LANE_DIR/llama.cpp/build/bin/llama-server; ASR_NAME=qwen3-asr-0.6b; ASR_CTX=8192
ASR_FILE=Qwen3-ASR-0.6B-Q8_0.gguf; MMPROJ_FILE=mmproj-Qwen3-ASR-0.6B-Q8_0.gguf
KOKORO_MODEL_FILE=kokoro-v1.0.onnx; KOKORO_VOICES_FILE=voices-v1.0.bin
MEM_HIGH=4G; MEM_MAX=5G; MEM_NOTE="about 2.7 GB with the 0.6B"
"""


class TheRenders(unittest.TestCase):
    def render(self, template):
        code, out, err = bash(RENDER_ENV + function("render") + f'\nrender "{REPO}/{template}"')
        self.assertEqual(code, 0, err)
        return out

    def test_no_placeholder_is_left(self):
        for t in ("qwen38-speech.service.template", "qwen38-speech-launch.sh.template"):
            with self.subTest(t):
                self.assertEqual(re.findall(r"__[A-Z][A-Z0-9_]*__", self.render(t)), [])

    def test_every_placeholder_of_the_templates_is_one_render_fills(self):
        filled = set(re.findall(r'-e "s\|(__[A-Z0-9_]+__)\|', function("render")))
        used = set(re.findall(r"__[A-Z][A-Z0-9_]*__", UNIT_T + LAUNCH_T))
        self.assertEqual(used - filled, set())

    def test_the_unit_keeps_off_the_gpu_and_the_big_cores(self):
        unit = self.render("qwen38-speech.service.template")
        self.assertIn("Environment=CUDA_VISIBLE_DEVICES=\n", unit)
        self.assertIn("CPUAffinity=0 1 2 3 4 10 11 12 13 14\n", unit)
        self.assertIn("MemoryMax=5G\n", unit)
        self.assertIn("PrivateTmp=yes\n", unit)                # uploads and espeak copies go with it
        self.assertNotRegex(unit, r"(?m)^Conflicts=")   # it runs beside any lane
        self.assertIn("ExecStart=/bin/bash /home/u/.config/qwen38/launch-speech.sh\n", unit)

    def test_the_launcher_hides_the_gpu_and_splits_the_cores(self):
        launch = self.render("qwen38-speech-launch.sh.template")
        self.assertIn("export CUDA_VISIBLE_DEVICES=\n", launch)
        self.assertLess(launch.index("export CUDA_VISIBLE_DEVICES="), launch.index("taskset"))
        self.assertIn("taskset -c 0,1,2,3,4 /home/u/.local/share/qwen38-speech/llama.cpp/build/bin/llama-server", launch)
        self.assertIn("taskset -c 10,11,12,13,14 /home/u/.local/share/qwen38-speech/venv/bin/python", launch)
        self.assertIn("--host 127.0.0.1 --port 31001", launch)          # the recognizer: loopback only
        self.assertIn("-ngl 0 -t 5 -np 1 -c 8192", launch)
        self.assertIn("--asr-ctx 8192", launch)                         # the routes know where it ends
        self.assertIn("--no-webui", launch)
        self.assertIn("--cache-ram 0", launch)                          # no prompt cache of up to 8 GiB
        self.assertIn('wait -n "$ASR" "$FRONT"', launch)

    def test_an_update_reads_the_routes_port_and_bind_back_not_the_recognizers(self):
        launch = self.render("qwen38-speech-launch.sh.template").replace("--host 127.0.0.1 --port 31000",
                                                                          "--host 10.0.0.5 --port 31000")
        fd, name = tempfile.mkstemp(suffix=".sh")
        tmp = pathlib.Path(name)
        self.addCleanup(tmp.unlink)
        with open(fd, "w") as f:
            f.write(launch)
        code, out, err = bash(f'LAUNCHER="{tmp}"\n' + function("launcher_flag") +
                              '\necho "$(launcher_flag --port) $(launcher_flag --host) $(launcher_flag --asr-name) $(launcher_flag --default-voice)"')
        self.assertEqual(code, 0, err)
        self.assertEqual(out.split(), ["31000", "10.0.0.5", "qwen3-asr-0.6b", "af_heart"])


class TheGuards(unittest.TestCase):
    def test_root_is_refused_and_tools_are_checked_first(self):
        self.assertIn('if [ "$(id -u)" = "0" ] && [ "${ALLOW_ROOT:-0}" != "1" ]', SCRIPT)
        self.assertIn("for t in git curl cmake c++ sha256sum taskset; do", SCRIPT)
        self.assertIn("python3 -c 'import venv, ensurepip'", SCRIPT)
        self.assertIn('command -v ffmpeg >/dev/null || die', SCRIPT)

    def test_the_build_never_asks_for_cuda(self):
        self.assertIn("-DGGML_CUDA=OFF", SCRIPT)
        self.assertNotRegex(SCRIPT, r"GGML_CUDA=ON")

    def test_it_is_not_started_at_boot_unless_asked(self):
        self.assertIn('if [ "${SPEECH_BOOT:-0}" = "1" ]; then sudo systemctl enable', SCRIPT)
        self.assertEqual(len(re.findall(r"systemctl enable", SCRIPT)), 1)

    def test_the_proof_leaves_the_lane_as_it_found_it(self):
        self.assertIn('if [ "$KEEP" -eq 0 ]; then sudo systemctl stop "$UNIT"', SCRIPT)
        self.assertIn('"<asr_text>" not in heard', SCRIPT)

    def uninstall(self, with_models):
        """The --uninstall branch as written, against a lane folder of the test's own, with
        no unit installed (so no sudo is asked for)."""
        block = SCRIPT[SCRIPT.index('if [ "$ACTION" = uninstall ]; then'):SCRIPT.index('step "1/6 Preflight"')]
        d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(d)])
        lane = d / "lane"
        for sub in ("venv/bin", "llama.cpp/build/bin", "models"):
            (lane / sub).mkdir(parents=True)
        (lane / "llama.cpp" / "qwen38-build.log").write_text("log")
        (lane / "speech-server.py").write_text("routes")
        if with_models:
            (lane / "models" / "Qwen3-ASR-0.6B-Q8_0.gguf").write_text("weights")
        else:
            (lane / "models").rmdir()
        launcher = d / "launch-speech.sh"
        launcher.write_text("launcher")
        env = (f"ACTION=uninstall; UNIT=qwen38-speech.service; INSTALLED={d}/no-unit; LAUNCHER={launcher}\n"
               f"LANE_DIR={lane}; VENV={lane}/venv; LLAMA_DIR={lane}/llama.cpp; MODELS={lane}/models\n"
               "step(){ :; }\nsudo(){ echo \"sudo $*\" >&2; return 1; }\n")
        code, out, err = bash(env + block)
        self.assertEqual(code, 0, err)
        self.assertNotIn("sudo", err)
        self.assertFalse(launcher.exists())
        return lane, out

    def test_the_uninstall_keeps_the_models_and_says_how_to_reclaim_them(self):
        lane, out = self.uninstall(with_models=True)
        self.assertEqual(sorted(p.name for p in lane.iterdir()), ["models"])
        self.assertTrue((lane / "models" / "Qwen3-ASR-0.6B-Q8_0.gguf").exists())
        self.assertIn(f"rm -rf {lane}/models", out)

    def test_the_uninstall_without_models_leaves_nothing(self):
        lane, out = self.uninstall(with_models=False)
        self.assertFalse(lane.exists(), out)

    def test_the_voice_check_speaks_the_languages_the_routes_speak(self):
        # the installer refuses a SPEECH_VOICE the routes would refuse at every request
        import importlib.util
        spec = importlib.util.spec_from_file_location("speech_server", REPO / "speech-server.py")
        ss = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ss)
        served = re.search(r'if v\[:1\] in "([a-z]+)"\]', SCRIPT).group(1)
        self.assertEqual(sorted(served), sorted(ss.LANG_OF_PREFIX))

    def test_the_script_parses(self):
        r = subprocess.run(["bash", "-n", str(REPO / "install-speech.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
