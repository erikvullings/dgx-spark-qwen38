#!/usr/bin/env python3
"""uninstall.sh --list names installed.json (2026-10-17).

Since v1.36.0 every run of install.sh that ends well writes ~/.config/qwen38/installed.json:
the release it installed and the choices it recorded. The inventory of the config folder named
each file in it that is not obvious (the Agent view's credentials, the markers of a lane left
out), and the record joins them: it goes with the folder on --yes, and a folder kept keeps the
box's age for the next install. The block runs here as uninstall.sh has it.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "uninstall.sh").read_text()
START = TEXT.index('if [ -d "$CONFIG_DIR" ]; then\n  echo "  config    $CONFIG_DIR')
BLOCK = TEXT[START:TEXT.index("\nfi\n", START) + 4]


def inventory(files):
    cfg = pathlib.Path(tempfile.mkdtemp(prefix="uninst-cfg-"))
    for name in files:
        (cfg / name).write_text("{}\n")
    script = 'set -euo pipefail\ndir_size(){ echo 4K; }\n' + f'CONFIG_DIR="{cfg}"\n' + BLOCK + "echo INVENTORY-END\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    return cfg, r.stdout + r.stderr


class TheInventoryNamesTheRecord(unittest.TestCase):
    def test_a_box_with_a_record_is_told_what_it_is(self):
        cfg, out = inventory(["api-key", "installed.json"])
        self.assertIn(f"  record    {cfg}/installed.json (the release the last ./install.sh installed, "
                      "and the choices it recorded)", out)
        self.assertIn("INVENTORY-END", out)

    def test_a_box_without_one_gets_no_such_line(self):
        _, out = inventory(["api-key"])
        self.assertNotIn("installed.json", out)
        self.assertIn("INVENTORY-END", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
