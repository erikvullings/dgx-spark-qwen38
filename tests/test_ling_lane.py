#!/usr/bin/env python3
"""Ling staging must preserve its own pinned SGLang recipe and exclusive startup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

R = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class LingRecipe(unittest.TestCase):
    def test_optimisations_and_memory_match_the_cookbook(self):
        m = module('ling_launch_test', R / 'ling-launch.py')
        values = m.ARGS
        def value(flag):
            return values[values.index(flag) + 1]
        self.assertEqual(value('--mem-fraction-static'), '0.68')
        self.assertEqual(value('--moe-runner-backend'), 'humming')
        self.assertEqual(value('--speculative-num-steps'), '3')
        self.assertEqual(value('--speculative-num-draft-tokens'), '4')
        self.assertEqual(value('--max-running-requests'), '1')
        self.assertEqual(value('--context-length'), '262144')
        self.assertEqual(value('--model-path'), value('--speculative-draft-model-path'))
        self.assertEqual(m.ENV['SGLANG_ENABLE_FP8_LM_HEAD'], '1')
        self.assertNotIn('--api-key', m.ARGS)
        self.assertEqual(value('--config'), '/out/engine-secrets.yaml')

    def test_staged_unit_keeps_auth_and_compilation_before_serving(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = home / '.config/qwen38'
            cfg.mkdir(parents=True)
            hf = home / 'hf'
            snap = hf / 'hub/models--inclusionAI--Ling-3.0-flash-fp4/snapshots/3bae1cf4011b48475b2cc038fff283af49053ebc'
            snap.mkdir(parents=True)
            (snap / 'tokenizer_config.json').write_text(json.dumps({'chat_template': '{{ messages }}'}))
            out = home / 'unit'
            run = subprocess.run(['bash', str(R / 'install-ling.sh'), '--stage', str(out)], capture_output=True,
                                 text=True, env={**os.environ, 'HOME': str(home), 'ALLOW_ROOT': '1',
                                                 'QWEN38_MEMORY_CLASS': '128', 'HF_CACHE': str(hf)}, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            text = out.read_text()
            self.assertNotRegex(text, r'__[A-Z][A-Z0-9_]*__')
            self.assertLess(text.index('ling-precompile.py'), text.index('ExecStart='))
            self.assertIn('engine-secrets.yaml,target=/out/engine-secrets.yaml,readonly', text)
            self.assertIn('ExecStopPost=', text)
            self.assertIn('qwen38-nanbeige.service', text)

    def test_precompile_has_bounded_wait_and_no_global_killing(self):
        text = (R / 'ling-precompile.py').read_text()
        self.assertIn('os.killpg(proc.pid', text)
        self.assertIn('timeout=900', text)
        self.assertNotIn('pkill ', text)
        self.assertLess(text.index('if live_group():'), text.index('subprocess.run([ninja'))


class KernelPreparation(unittest.TestCase):
    def test_a_cached_kernel_does_not_load_any_model(self):
        m = module('ling_precompile_cached', R / 'ling-precompile.py')
        with patch.object(m, 'find', return_value=Path('/cache/kernel.so')), patch.object(m.subprocess, 'Popen') as spawn:
            m.main()
            spawn.assert_not_called()

    def test_a_missing_compiler_refuses_before_model_loading(self):
        m = module('ling_precompile_compiler', R / 'ling-precompile.py')
        with patch.object(m, 'find', return_value=None), patch.object(m.shutil, 'which', return_value=None), patch.object(m.subprocess, 'Popen') as spawn:
            with self.assertRaises(SystemExit):
                m.main()
            spawn.assert_not_called()

    def test_a_failed_rule_generation_never_starts_ninja(self):
        m = module('ling_precompile_failure', R / 'ling-precompile.py')
        child = Mock()
        child.poll.return_value = 1
        with patch.object(m, 'find', return_value=None), patch.object(m.shutil, 'which', return_value='/bin/ninja'), patch.object(m.subprocess, 'Popen', return_value=child), patch.object(m.subprocess, 'run') as compile_kernel:
            with self.assertRaises(SystemExit):
                m.main()
            compile_kernel.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
