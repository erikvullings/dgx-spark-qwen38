#!/usr/bin/env python3
"""Groot must stage an mixed-precision Llama unit, never the inherited hybrid/MTP recipe."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
TEXT = (REPO / 'install-groot.sh').read_text()
PINS = dict(re.findall(r'^(GROOT_\w+)="([^"]+)"', TEXT, re.M))


class GrootStage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / 'home'
        self.cfg = self.home / '.config/qwen38'
        self.cfg.mkdir(parents=True)
        self.cache = self.root / 'hf'
        self.snap = self.cache / 'hub' / ('models--' + PINS['GROOT_REPO'].replace('/', '--')) / 'snapshots' / PINS['GROOT_REV']
        self.snap.mkdir(parents=True)
        self.template = '{% for m in messages %}{{ m.content }}{% endfor %}'
        (self.snap / 'chat_template.jinja').write_text(self.template)
        self.out = self.root / 'unit.service'

    def stage(self, memory='128'):
        return subprocess.run(['bash', str(REPO / 'install-groot.sh'), '--stage', str(self.out)],
                              env={'PATH': '/usr/bin:/bin', 'HOME': str(self.home), 'ALLOW_ROOT': '1',
                                   'QWEN38_MEMORY_CLASS': memory, 'HF_CACHE': str(self.cache),
                                   'LANES_UNIT_DIR': str(self.root / 'units')},
                              text=True, capture_output=True, timeout=20)

    def test_stage_preserves_pinned_template_and_has_no_hybrid_flags(self):
        run = self.stage()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        unit = self.out.read_text()
        self.assertNotRegex(unit, r'__[A-Z][A-Z0-9_]*__')
        self.assertIn('--revision ' + PINS['GROOT_REV'], unit)
        self.assertIn('--dtype bfloat16', unit)
        self.assertIn('--max-model-len 131072', unit)
        self.assertIn('--tool-call-parser qwen3_coder', unit)
        self.assertNotIn('--speculative', unit)
        self.assertNotIn('--quantization', unit)
        self.assertNotIn('--hf-overrides', unit)
        self.assertNotIn('--load-format b12x', unit)
        self.assertNotIn('source=' + str(self.cfg) + ',', unit)
        tpl = self.cfg / 'chat-template-groot.jinja'
        self.assertEqual(tpl.read_text(), self.template)
        before = tpl.stat().st_mtime_ns
        self.assertEqual(self.stage().returncode, 0)
        self.assertEqual(tpl.stat().st_mtime_ns, before)

    def test_missing_template_cannot_stage_a_broken_unit(self):
        (self.snap / 'chat_template.jinja').unlink()
        run = self.stage()
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(self.out.exists())

    def test_64gb_is_refused_before_staging(self):
        run = self.stage('64')
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('128 GB', run.stdout + run.stderr)
        self.assertFalse(self.out.exists())

    def test_tokenizer_config_template_fallback(self):
        (self.snap / 'chat_template.jinja').unlink()
        (self.snap / 'tokenizer_config.json').write_text(json.dumps({'chat_template': self.template}))
        run = self.stage()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


class GrootProvider(unittest.TestCase):
    def test_text_only_provider_exposes_thinking_and_preserves_agent_history(self):
        spec = importlib.util.spec_from_file_location('groot_provider', REPO / 'oc-groot.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        p = module.provider('http://localhost:30001/v1', '/tmp/key', 28000, 4000, 'Groot')
        model = p['models']['groot-65b-a6b']
        self.assertEqual(set(model['variants']), {'low', 'medium', 'xhigh', 'none'})
        self.assertFalse(model['attachment'])
        self.assertEqual(model['modalities']['input'], ['text'])
        self.assertTrue(model['variants']['xhigh']['chat_template_kwargs']['preserve_thinking'])
        self.assertFalse(model['variants']['none']['chat_template_kwargs']['enable_thinking'])

    def test_pi_generator_includes_groot_and_its_native_tiers(self):
        spec = importlib.util.spec_from_file_location('groot_pi_gen', REPO / 'pi-gen.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        artifact = {"model": "groot/groot-65b-a6b", "provider": {"groot": {
            "options": {"baseURL": "http://localhost:30001/v1", "apiKey": "test"},
            "models": {"groot-65b-a6b": {"limit": {"context": 100000, "output": 16000},
                       "variants": {"low": {}, "medium": {}, "xhigh": {}, "none": {}}}}}}}
        models, settings, efforts = module.build(artifact)
        model = models['providers']['groot']['models'][0]
        self.assertTrue(model['compat']['supportsReasoningEffort'])
        self.assertEqual(model['thinkingLevelMap']['off'], 'none')
        self.assertEqual(settings['defaultProvider'], 'groot')
        self.assertEqual(settings['modelThinkingLevels']['groot/groot-65b-a6b'], 'xhigh')


if __name__ == '__main__':
    unittest.main(verbosity=2)
