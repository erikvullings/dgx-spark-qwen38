#!/usr/bin/env python3
"""What the cockpit's ergonomics review found (2026-10-15), each on the page as it runs: Library
opening empty below its recipes, a median that was not one, "no engine" over a serving Kolibri
lane, the Kolibri engine's polls drowning its own journal, and Decide asking for a text lane
while one served. Same harness as test_page_behaviour.py."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_page_behaviour import run  # noqa: E402

REG = {"images": [], "other_models": [], "caches": ["/c"], "models": [],
       "targets": [{"target": "stock", "repo": "RadixArk/Qwen3.8-27B-NVFP4", "rev": "a" * 40, "state": "complete", "installed": True},
                   {"target": "kolibri-nvfp4", "repo": "Quantinine/Kolibri-1-NVFP4A16-FP8", "rev": "b" * 40, "state": "absent", "installed": True}]}


class LibraryFillsItself(unittest.TestCase):
    """The checkpoints, what each lane can load (and its Download), the engine images and the
    other models waited for a click on Scan, while opening the view had just made the server
    read them (the recipes' load refreshes the registry): it opened empty below its recipes."""

    def open_library(self, times=1):
        return run(self, r"""
        __fetch = async (url, init) => url.startsWith('/api/recipes') ? __response(200, {builtin: [], custom: [], custom_dir: ''})
          : url.startsWith('/api/registry') ? __response(200, %s) : new Promise(() => {});
        feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-sglang.service': eng('ready')})});
        for (let i = 0; i < %d; i++){ showView('library'); await __settle(); await __settle(); showView('now'); await __settle(); }
        showView('library'); await __settle();
        report({reads: __fetches.map(f => f.url).filter(u => u.startsWith('/api/registry')),
                targets: $('reg-targets').tBodies[0].rows.length});
        """ % (json.dumps(REG), times))

    def test_it_shows_what_the_server_read_as_it_opens(self):
        r = self.open_library()
        self.assertEqual(r["targets"], 2)
        self.assertEqual(r["reads"], ["/api/registry"])      # the server's read, not a second scan

    def test_it_reads_once_per_page_and_scan_still_scans(self):
        self.assertEqual(self.open_library(times=3)["reads"], ["/api/registry"])
        r = run(self, r"""
        __fetch = async (url, init) => url.startsWith('/api/registry') ? __response(200, %s) : new Promise(() => {});
        $('reg-btn').click(); await __settle();
        report(__fetches.map(f => f.url).filter(u => u.startsWith('/api/registry')));
        """ % json.dumps(REG))
        self.assertEqual(r, ["/api/registry?refresh=1"])


class TheMedianOfAnEvenCount(unittest.TestCase):
    """"Last 2 boots: 14 min 47, 15 min 37. Median 15 min 37": the upper of the two middle
    values stood for the median, and for every "answers in about" said from it."""

    def test_an_even_count_takes_the_mean_of_its_two_middles(self):
        r = run(self, r"""
        feed({config: CONFIG, units: UNITS, lifecycle: life({
          'qwen38-flash.service': eng('stopped', {target: 'flash', boots: [937, 887]}),
          'qwen38-image.service': eng('stopped', {target: 'image', boots: [61, 75, 60]}),
          'qwen38-video.service': eng('stopped', {target: 'video', boots: [700, 720, 712, 735]})})});
        report({two: bootSeconds('qwen38-flash.service'), three: bootSeconds('qwen38-image.service'),
                four: bootSeconds('qwen38-video.service'), said: readyIn('qwen38-flash.service')});
        """)
        self.assertEqual(r["two"], 912)
        self.assertEqual(r["three"], 61)
        self.assertEqual(r["four"], 716)
        self.assertIn("15 min 12", r["said"])


class TheRequestIdOverrideOnKolibri(unittest.TestCase):
    """Traffic said "Request id override: no engine" while Kolibri served: the override is an
    SGLang setting the guard reads on the SGLang lanes only."""

    def test_a_serving_kolibri_lane_is_not_no_engine(self):
        r = run(self, r"""
        const guard = {lane: null, override: null, state: 'ok', verdict: 'no abandoned request in this window', window: '10m', guard: {}, zombies: {}};
        feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-kolibri.service': eng('ready', {target: 'kolibri'})})});
        feed({reqguard: guard});
        const kolibri = txt('zg-override');
        feed({lifecycle: life({'qwen38-sglang.service': eng('stopped')})});
        feed({reqguard: guard});
        report({kolibri, none: txt('zg-override')});
        """)
        self.assertIn("not on the Kolibri lane", r["kolibri"])
        self.assertEqual(r["none"], "no engine")


class TheEnginesOwnPollsAreDimmed(unittest.TestCase):
    """The Kolibri engine's journal was the box's own polls, line after line (its metrics every
    few seconds, its version, its models): the page dimmed the SGLang lanes' polls only."""

    def test_kolibris_polls_dim_like_the_others_and_a_request_does_not(self):
        r = run(self, r"""
        logLines = ['(APIServer pid=1) INFO:     127.0.0.1:1 - "GET /metrics HTTP/1.1" 200 OK',
                    '(APIServer pid=1) INFO:     127.0.0.1:2 - "GET /version HTTP/1.1" 200 OK',
                    '(APIServer pid=1) INFO:     127.0.0.1:3 - "GET /v1/models HTTP/1.1" 200 OK',
                    '(APIServer pid=1) INFO:     127.0.0.1:4 - "GET /health HTTP/1.1" 200 OK',
                    '(APIServer pid=1) INFO:     127.0.0.1:5 - "POST /v1/chat/completions HTTP/1.1" 200 OK'];
        drawLog();
        report([...$('log-view').querySelectorAll('span')].map(s => s.className || ''));
        """)
        self.assertEqual(r, ["l-dim", "l-dim", "l-dim", "l-dim", ""])


class SystemOneNamesTheLanesThatServeIt(unittest.TestCase):
    """Under Ask, Decide said System One "needs a text lane" while the Kolibri text lane served,
    over a note saying the 27B or the flash lane answers it."""

    def test_the_line_under_ask_names_them(self):
        r = run(self, r"""
        __fetch = async url => url === '/api/systemone' ? __response(200, {available: false, lane: '',
          reason: 'The Kolibri lane is serving: System One is answered by the 27B or the flash lane'}) : new Promise(() => {});
        feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-kolibri.service': eng('ready', {target: 'kolibri'})})});
        showView('decide'); await __settle();
        report(txt('s1-why'));
        """)
        self.assertIn("the 27B or the flash lane", r)
        self.assertNotIn("needs a text lane", r)


if __name__ == "__main__":
    unittest.main()
