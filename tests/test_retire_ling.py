"""An upgrade retires the old service while retaining user data."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

R = Path(__file__).resolve().parents[1]

class RetireLing(unittest.TestCase):
    def run_retire(self, fail_stop=False):
        with tempfile.TemporaryDirectory() as t:
            home = Path(t)
            cfg = home / '.config/qwen38'
            cfg.mkdir(parents=True)
            key = cfg / 'api-key'
            key.write_text('keep-key')
            cache = cfg / 'ling-cache'
            cache.mkdir()
            (cache / 'kernel').write_text('keep-cache')
            units = home / 'units'
            units.mkdir()
            unit = units / 'qwen38-ling.service'
            unit.write_text('[Service]\n')
            bin = home / 'bin'
            bin.mkdir()
            for name, body in {
                'sudo': 'exec "$@"',
                'docker': 'exit 0',
                'systemctl': 'case "$1" in stop) exit ' + ('1' if fail_stop else '0') + ';; is-enabled|is-active) exit 1;; *) exit 0;; esac',
            }.items():
                p = bin / name
                p.write_text('#!/bin/sh\n' + body + '\n')
                p.chmod(0o755)
            run = subprocess.run(['bash', str(R / 'retire-ling.sh')], capture_output=True, text=True,
                                 env={**os.environ, 'HOME': str(home), 'PATH': str(bin) + ':/usr/bin:/bin',
                                      'LANES_UNIT_DIR': str(units), 'ALLOW_ROOT': '1'}, timeout=20)
            self.assertEqual(key.read_text(), 'keep-key')
            self.assertEqual((cache / 'kernel').read_text(), 'keep-cache')
            return run, unit.exists()

    def test_removes_only_the_service_and_keeps_key_and_cache(self):
        run, remains = self.run_retire()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertFalse(remains)

    def test_failed_stop_does_not_remove_unit(self):
        run, remains = self.run_retire(True)
        self.assertNotEqual(run.returncode, 0)
        self.assertTrue(remains)
