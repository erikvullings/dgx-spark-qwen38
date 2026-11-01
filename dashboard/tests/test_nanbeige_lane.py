#!/usr/bin/env python3
"""Nanbeige must be managed as vLLM and never receive the Qwen-only control routes."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DASH = Path(__file__).resolve().parents[1]
U = 'qwen38-nanbeige.service'


class NanbeigeManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {'COCKPIT_DRY_RUN': '1', 'COCKPIT_CONFIG_DIR': cls.tmp.name,
                                        'COCKPIT_PORT': '0', 'COCKPIT_AGENT_PORT': '0'})
        cls.env.start()
        cls.old_path = list(sys.path)
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location('nanbeige_cockpit_test', DASH / 'cockpit.py')
        cls.ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ck)
        cls.unit = Path(cls.tmp.name) / U
        cls.unit.write_text('ExecStart=docker run eugr/spark-vllm-b12x@sha256:' + 'a' * 64 + '\n')
        cls.ck.NANBEIGE_UNIT_PATH = cls.unit

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls.old_path
        cls.env.stop()
        cls.tmp.cleanup()

    def setUp(self):
        self.ck.LIFE['states'] = {U: 'ready'}

    def test_lane_target_and_pin_are_consistent(self):
        self.assertEqual(self.ck.target_lane('nanbeige'), 'nanbeige')
        self.assertEqual(self.ck.target_unit('nanbeige'), self.unit)
        self.assertEqual(self.ck.TARGET_PIN['nanbeige'], 'NANBEIGE_REV')
        self.assertIn('nanbeige', self.ck.ACTIONS['switch']['params']['target'])
        self.assertIn(U, self.ck.lc.ENGINE_UNITS)
        self.assertNotIn(U, self.ck.lc.TEXT_UNITS)
        self.assertEqual(self.ck.NEMOTRON_LANES[U][3], 'nvfp4/fp8')
        self.assertIsNone(self.ck.B12X_SPEC[U])

    def test_prepared_checks_the_pinned_image(self):
        with patch.object(self.ck, 'run', return_value='sha256:' + 'b' * 64):
            self.assertTrue(self.ck.lane_prepared('nanbeige'))
        with patch.object(self.ck, 'run', return_value=''):
            self.assertFalse(self.ck.lane_prepared('nanbeige'))

    def test_sglang_only_actions_are_refused_with_the_correct_lane_name(self):
        for name in ('flush_cache', 'abort_all', 'fit_opencode'):
            self.assertIn('Nanbeige', self.ck.b12x_cannot(name))

    def test_systemone_is_explicitly_unavailable_on_this_tokenizer(self):
        out = self.ck.systemone_available()
        self.assertFalse(out['available'])
        self.assertIn('Nanbeige', out['reason'])

    def test_standard_vllm_boot_reader_is_used(self):
        with patch.object(self.ck, 'run', return_value='container'), \
             patch.object(self.ck, 'kolibri_healthy', return_value=False), \
             patch.object(self.ck, '_nemotron_log', return_value=['Initializing a V1 LLM engine',
                                                                'Application startup complete']):
            _, boot, running = self.ck.nemotron_engine_state(U, active='active', sub='running',
                                                            prev_state='starting', enter_key='nanbeige-test')
        self.assertTrue(running)
        self.assertTrue(boot['fired_up'])
        self.assertNotIn('MTP', boot.get('detail', ''))


if __name__ == '__main__':
    unittest.main(verbosity=2)
