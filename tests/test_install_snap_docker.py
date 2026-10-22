#!/usr/bin/env python3
"""Docker from the Snap package, beside the NVIDIA Container Toolkit that DGX OS installs on the host,
fails to start a container with the GPU (canonical/docker-snap#357). The installer pulled the image,
then its GPU check threw Docker's error away and said the toolkit was missing or unconfigured, with a
fix that did not apply (issue #58, 2026-10-21). The preflight now stops on that pair before the first
byte, and the check shows what Docker said, its exit status and the engine it reached.

The scripts' own lines run here against a fake docker and a fake dpkg-query first on the PATH: the
box's Docker is never asked."""
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
INSTALL = (REPO / "install.sh").read_text()
LINES = INSTALL.splitlines()
DIE = next(ln for ln in LINES if ln.startswith("die()"))
STEP = next(ln for ln in LINES if ln.startswith("step()"))
SNAP_ROOT = "/var/snap/docker/common/var-lib-docker"
SNAP_RUNTIME = "/snap/docker/3614/usr/bin/nvidia-container-runtime"
MOUNT_ERROR = ("docker: Error response from daemon: failed to create task for container: failed to create shim "
               "task: OCI runtime create failed: runc create failed: unable to start container process: error "
               "during container init: failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: "
               "no such file or directory")

DOCKER = r'''#!/bin/bash
# a Docker engine as the environment describes it: FAKE_ROOT, FAKE_RUNTIMES (name=path ...), FAKE_VERSION,
# FAKE_HOST, and what `docker run` prints (FAKE_RUN_OUT, FAKE_RUN_ERR) and exits with (FAKE_RUN_RC)
case "$1 $2" in
  "info --format")
    case "$3" in
      '{{.DockerRootDir}}') echo "$FAKE_ROOT" ;;
      '{{.ServerVersion}}') echo "$FAKE_VERSION" ;;
      '{{range $k, $v := .Runtimes}}{{$v.Path}} {{end}}') for r in $FAKE_RUNTIMES; do printf '%s ' "${r#*=}"; done; echo ;;
      '{{range $k, $v := .Runtimes}}{{$k}}={{$v.Path}} {{end}}') for r in $FAKE_RUNTIMES; do printf '%s ' "$r"; done; echo ;;
      *) exit 1 ;;
    esac ;;
  "context inspect") echo "$FAKE_HOST" ;;
  "run --rm") printf '%s' "$FAKE_RUN_OUT"; [ -z "$FAKE_RUN_ERR" ] || echo "$FAKE_RUN_ERR" >&2; exit "${FAKE_RUN_RC:-0}" ;;
  *) exit 1 ;;
esac
'''
DPKG = '''#!/bin/bash
# the host's toolkit packages, installed when FAKE_TOOLKIT=1
[ "${FAKE_TOOLKIT:-0}" = 1 ] || exit 1
printf 'install ok installed\\ninstall ok installed\\n'
'''


def engine(snap_root=False, snap_runtime=False, toolkit=True, run_out="", run_err="", run_rc=0):
    return {"FAKE_ROOT": SNAP_ROOT if snap_root else "/var/lib/docker",
            "FAKE_RUNTIMES": "io.containerd.runc.v2=runc nvidia=" + (SNAP_RUNTIME if snap_runtime else "nvidia-container-runtime"),
            "FAKE_VERSION": "29.8.0", "FAKE_HOST": "unix:///var/run/docker.sock", "FAKE_TOOLKIT": "1" if toolkit else "0",
            "FAKE_RUN_OUT": run_out, "FAKE_RUN_ERR": run_err, "FAKE_RUN_RC": str(run_rc)}


def run(script, env):
    with tempfile.TemporaryDirectory() as t:
        bin_ = pathlib.Path(t) / "bin"
        bin_.mkdir()
        for name, body in (("docker", DOCKER), ("dpkg-query", DPKG)):
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)
        r = subprocess.run(["bash", "-c", f'set -euo pipefail\n. "{REPO}/lanes.sh"\n{DIE}\n{STEP}\n{script}'],
                           capture_output=True, text=True, timeout=30,
                           env={"PATH": f"{bin_}:/usr/bin:/bin", "HOME": t, **env})
    return r.returncode, r.stdout, r.stderr


def block(test, start, stop):
    """install.sh's lines from the first that starts with `start` through the first after it with `stop`."""
    i = next((n for n, ln in enumerate(LINES) if ln.strip().startswith(start)), None)
    if i is None:
        test.fail(f"install.sh has no line starting with {start!r}")
    j = next(n for n in range(i, len(LINES)) if stop in LINES[n])
    return "\n".join(LINES[i:j + 1])


class TheEngineThisAccountReaches(unittest.TestCase):
    def test_the_snap_engine_is_known_by_its_data_or_its_runtime(self):
        for kw, said in (({"snap_root": True}, f"its data in {SNAP_ROOT}"),
                         ({"snap_runtime": True}, f"a runtime at {SNAP_RUNTIME}")):
            with self.subTest(kw):
                self.assertEqual(run("docker_snap_engine", engine(**kw)), (0, said + "\n", ""))
        rc, out, _ = run("docker_snap_engine", engine())
        self.assertEqual((rc, out), (1, ""), "Docker CE is not the Snap engine")

    def test_the_host_toolkit_is_read_from_its_packages(self):
        self.assertEqual(run("host_nvidia_toolkit", engine(toolkit=True))[0], 0)
        self.assertEqual(run("host_nvidia_toolkit", engine(toolkit=False))[0], 1)


class ThePreflight(unittest.TestCase):
    def check(self):
        return block(self, 'if SNAP_DOCKER="$(docker_snap_engine)" && host_nvidia_toolkit; then', "fi")

    def test_the_snap_engine_beside_the_host_toolkit_stops_before_the_first_byte(self):
        rc, out, err = run(self.check() + '\necho "went on"', engine(snap_root=True))
        self.assertEqual(rc, 1)
        self.assertNotIn("went on", out)
        for said in ("Snap package", SNAP_ROOT, "https://github.com/canonical/docker-snap#nvidia-container-toolkit-compatibility",
                     "Nothing was changed", "sudo snap stop --disable docker", "sudo systemctl enable --now docker"):
            self.assertIn(said, err)

    def test_any_other_engine_goes_on(self):
        for kw in ({}, {"snap_root": True, "toolkit": False}):
            with self.subTest(kw):
                self.assertEqual(run(self.check() + '\necho "went on"', engine(**kw))[:2], (0, "went on\n"))


class TheGpuCheck(unittest.TestCase):
    # from its step line to its last: the lines v1.38.1 ran and the ones after it alike
    def check(self, **kw):
        code = block(self, 'step "3/10 Verifying the container can see the GPU"', 'echo "OK, container sees: $GPU_SEEN"')
        return run('PULLED_IMAGE="lmsysorg/sglang@sha256:' + "b" * 64 + '"\n' + code, engine(**kw))

    def test_a_container_that_sees_the_gpu_says_it(self):
        rc, out, err = self.check(run_out="GPU 0: NVIDIA GB10 (UUID: GPU-0)\n")
        self.assertEqual((rc, out.splitlines()[-1]), (0, "OK, container sees: GPU 0: NVIDIA GB10 (UUID: GPU-0)"), err)

    def test_the_snap_engine_shows_dockers_own_error_and_names_itself(self):
        rc, out, err = self.check(snap_root=True, toolkit=False, run_err=MOUNT_ERROR, run_rc=125)
        self.assertEqual(rc, 1)
        for said in ("exited 125", "nvidia-cuda-mps-control: no such file or directory", "Engine: Docker 29.8.0, data in " + SNAP_ROOT,
                     "Snap package", "https://github.com/canonical/docker-snap#nvidia-container-toolkit-compatibility"):
            self.assertIn(said, err)
        self.assertNotIn("nvidia-ctk runtime configure", err, "no fix that does not apply")

    def test_any_other_failure_shows_it_and_names_the_toolkit_as_a_possible_cause(self):
        err_line = 'docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].'
        rc, out, err = self.check(run_err=err_line, run_rc=125)
        self.assertEqual(rc, 1)
        for said in ("exited 125", err_line, "Engine: Docker 29.8.0, data in /var/lib/docker, endpoint unix:///var/run/docker.sock",
                     "If the NVIDIA Container Toolkit is missing or unconfigured: sudo nvidia-ctk runtime configure"):
            self.assertIn(said, err)
        self.assertIsNone(re.search(r"The NVIDIA Container Toolkit is missing or unconfigured\. Fix", err), "a cause, not the diagnosis")


if __name__ == "__main__":
    unittest.main()
