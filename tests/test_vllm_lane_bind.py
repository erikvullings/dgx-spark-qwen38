#!/usr/bin/env python3
"""A vLLM lane's engine answers on the bind the text lanes use, theirs read first (2026-10-17).

vllm-lane.sh read the bind in the lane's own unit before the text lanes': a Kolibri or Nemotron
unit rendered beside a text lane on every interface (the default before v1.17.0) kept 0.0.0.0
after an update of v1.36.0 or later moved that text lane to 127.0.0.1, whenever the lane was not
in that update's run (left out, then brought back). Its own unit is read only on a box with no
text lane installed, and ENGINE_BIND, when set, wins over both. The function runs here as the
installers source it.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


def bind(text_host=None, lane_host=None, env_bind=None, flash_host=None):
    d = pathlib.Path(tempfile.mkdtemp(prefix="vlane-bind-"))
    units, cfg = d / "units", d / "cfg"
    units.mkdir(); cfg.mkdir()
    if text_host:
        (units / "qwen38-sglang.service").write_text(
            f"[Service]\nExecStart=/usr/bin/docker run x --model-path m --host {text_host} --port 30000\n")
    if flash_host:
        (cfg / "launch-flash.sh").write_text(f"exec docker run y --host {flash_host} --port 30000\n")
    if lane_host:
        (units / "qwen38-kolibri.service").write_text(
            f"[Service]\nExecStart=/usr/bin/docker run z vllm serve --host {lane_host} --port 30000\n")
    script = ('set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n'
              f'. "{REPO}/lanes.sh"\n. "{REPO}/vllm-lane.sh"\n'
              f'LANES_UNIT_DIR="{units}"; CONFIG_DIR="{cfg}"; INSTALLED="{units}/qwen38-kolibri.service"\n'
              + (f'ENGINE_BIND={env_bind}\n' if env_bind else 'unset ENGINE_BIND\n')
              + 'vlane_engine_env ""\necho "BIND=$ENGINE_BIND"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(d)})
    return (r.stdout + r.stderr).strip().splitlines()[-1]


class TheLaneFollowsTheTextLanes(unittest.TestCase):
    def test_a_text_lane_moved_to_loopback_takes_the_lane_with_it(self):
        self.assertEqual(bind(text_host="127.0.0.1", lane_host="0.0.0.0"), "BIND=127.0.0.1")

    def test_the_flash_launcher_counts_as_a_text_lane(self):
        self.assertEqual(bind(flash_host="127.0.0.1", lane_host="0.0.0.0"), "BIND=127.0.0.1")

    def test_a_text_lane_on_the_network_takes_it_there_too(self):
        self.assertEqual(bind(text_host="0.0.0.0", lane_host="127.0.0.1"), "BIND=0.0.0.0")

    def test_its_own_unit_only_with_no_text_lane(self):
        self.assertEqual(bind(lane_host="0.0.0.0"), "BIND=0.0.0.0")

    def test_engine_bind_set_wins(self):
        self.assertEqual(bind(text_host="127.0.0.1", lane_host="0.0.0.0", env_bind="0.0.0.0"), "BIND=0.0.0.0")

    def test_nothing_installed_is_loopback(self):
        self.assertEqual(bind(), "BIND=127.0.0.1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
