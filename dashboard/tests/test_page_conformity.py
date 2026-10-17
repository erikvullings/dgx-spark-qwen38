"""The page says, in orange, what the next update moves on this box (v1.36.0).

"Nobody behind in silence" (2026-10-16): a value an older release installed by default stayed on
a box for good, said once in an update's output, and the cockpit, the one place an operator
looks, said nothing. The update tier carries the list install.sh prints before its first step;
the banner strip says in orange what the next update moves, Settings lists each difference with
why it stays or the variable that keeps it, and neither a box as recommended nor one that
differs by its operator's choices alone gets a banner (Hasan, 2026-10-17: a choice is not a
delay, and a box that keeps one wore the banner for good). The update banner no longer promises
that an update keeps the context mode: an old default moves.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_page_behaviour import css_rules, run  # noqa: E402

MOVES = {"group": "moves", "knob": "ENGINE_BIND", "value": "0.0.0.0", "recommended": "127.0.0.1",
         "why": "box-before", "old": "v1.17.0"}
YOURS = {"group": "yours", "knob": "target.flash", "value": "flash-nvda", "recommended": "flash",
         "why": "recorded", "old": None}
SITE = {"group": "site", "knob": "HF_CACHE", "value": "/mnt/ssd/huggingface",
        "recommended": "/home/u/.cache/huggingface", "why": "site", "old": None}


def page(test, items, extra="{}"):
    conf = "null" if items is None else '{release: "v1.36.0", lane: "27b", items: %s}' % items
    return run(test, r"""
    const now = Date.now() / 1000;
    apply({config: {data: CONFIG, ts: now}, units: {data: UNITS, ts: now},
           update: {data: Object.assign({installed: 'v1.36.0'}, %s === null ? {} : {conformity: %s}, %s), ts: now},
           lifecycle: {data: life({'qwen38-sglang.service': eng('ready')}), ts: now}});
    showView('settings');
    const banner = [...$('banners').children].map(b => ({cls: b.className, lamp: b.querySelector('.lamp').className, text: b.textContent}));
    report({banners: banner, cap: txt('cf-cap'), capCls: $('cf-cap').className, head: txt('cf-head'),
            list: [...$('cf-list').children].map(li => ({cls: li.className, text: li.textContent}))});
    """ % (conf, conf, extra))


def js(items):
    import json
    return json.dumps(items)


class TheBannerSaysWhatTheNextUpdateMoves(unittest.TestCase):
    def test_an_old_default_is_said_first_with_what_the_next_update_does(self):
        r = page(self, js([MOVES, YOURS]))
        mine = [b for b in r["banners"] if "older release" in b["text"]]
        self.assertEqual(len(mine), 1, r["banners"])
        self.assertEqual(mine[0]["cls"], "banner orange")
        self.assertEqual(mine[0]["lamp"], "lamp orange")
        self.assertIn("This box runs a default of an older release.", mine[0]["text"])
        self.assertIn("ENGINE_BIND 0.0.0.0: the next update moves it to what v1.36.0 recommends, and keeps one "
                      "setting of yours. Settings lists each difference, and the variable that keeps it.", mine[0]["text"])

    def test_choices_alone_make_no_banner_and_settings_lists_them(self):
        r = page(self, js([YOURS, SITE]))
        self.assertEqual([b for b in r["banners"] if "orange" in b["cls"]], [], r["banners"])
        self.assertEqual(r["cap"], "2 differ")
        self.assertEqual([li["cls"] for li in r["list"]], ["cf yours", "cf site"])

    def test_a_box_as_recommended_gets_no_banner(self):
        r = page(self, js([]))
        self.assertEqual([b for b in r["banners"] if "recommends" in b["text"]], [])
        self.assertEqual(r["cap"], "as recommended")
        self.assertIn("Every setting this box serves is what v1.36.0 recommends.", r["head"])

    def test_nothing_known_is_nothing_said(self):
        r = page(self, None)
        self.assertEqual([b for b in r["banners"] if "recommends" in b["text"]], [])
        self.assertEqual(r["cap"], "unknown")

    def test_the_update_banner_no_longer_promises_to_keep_everything(self):
        r = page(self, js([]), extra="{latest: 'v1.36.1', behind: true}")
        upd = [b["text"] for b in r["banners"] if "is out" in b["text"]]
        self.assertEqual(len(upd), 1, r["banners"])
        self.assertNotIn("It keeps your target, context mode, port and cache", upd[0])
        self.assertIn("It keeps your choices, and moves a default an older release left to today's: it lists both "
                      "before its first step.", upd[0])


class TheServicesBehindTheirClone(unittest.TestCase):
    """installed.json names what the last run that ended well installed: the services can run
    behind their own clone, and the strip says why and the command that brings them up."""

    def banner(self, extra):
        r = page(self, js([]), extra=extra)
        return [b for b in r["banners"] if "install" in b["text"].lower() and "is out" not in b["text"]]

    def test_a_git_pull_with_no_install(self):
        b = self.banner("{installed: 'v1.36.0', clone: 'v1.36.1', behind_clone: true, behind_clone_why: 'commit'}")
        self.assertEqual(len(b), 1, b)
        self.assertEqual(b[0]["cls"], "banner orange")
        self.assertIn("The clone moved on since the last install: the services run v1.36.0, the clone is v1.36.1.", b[0]["text"])
        self.assertIn("A git pull changes the files, not what runs.", b[0]["text"])

    def test_a_run_stopped_at_no_start(self):
        b = self.banner("{behind_clone: true, behind_clone_why: 'no-start'}")
        self.assertIn("The last install stopped at --no-start.", b[0]["text"])

    def test_a_box_no_run_recorded(self):
        b = self.banner("{installed: 'before v1.36.0', behind_clone: true, behind_clone_why: 'no-record'}")
        self.assertIn("No install of v1.36.0 or later recorded what this box runs.", b[0]["text"])

    def test_services_that_follow_their_clone_get_none(self):
        self.assertEqual(self.banner("{behind_clone: false}"), [])


class SettingsListsEachDifference(unittest.TestCase):
    def test_each_one_with_why_it_stays_or_how_to_keep_it_what_moves_first(self):
        r = page(self, js([SITE, YOURS, MOVES]))
        self.assertEqual(r["cap"], "3 differ")
        self.assertEqual(r["capCls"], "cap orange")
        self.assertEqual([li["cls"] for li in r["list"]], ["cf moves", "cf yours", "cf site"])
        self.assertEqual(r["list"][0]["text"], "ENGINE_BIND 0.0.0.0: moves at the next update, the default until v1.17.0 "
                                               "(recommended: 127.0.0.1). To keep it: ENGINE_BIND=0.0.0.0 ./install.sh")
        self.assertEqual(r["list"][1]["text"], "flash target flash-nvda: stays, your choice, recorded as your choice "
                                               "(recommended: flash).")
        self.assertEqual(r["list"][2]["text"], "HF_CACHE /mnt/ssd/huggingface: stays, a setting of this site, updates "
                                               "never move it (recommended: /home/u/.cache/huggingface).")

    def test_the_drafter_points_at_the_line_that_keeps_it(self):
        drafter = dict(MOVES, knob="DRAFT2", value="z-lab/Qwen3.8-27B-DFlash2 (unquant, D=8)",
                       recommended="maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal (modelopt_fp4, D=16)", old="v1.9.0")
        r = page(self, js([drafter]))
        self.assertTrue(r["list"][0]["text"].startswith("27B drafter z-lab/Qwen3.8-27B-DFlash2 (unquant, D=8): moves"))
        self.assertTrue(r["list"][0]["text"].endswith("The update prints the variables that keep it."))


class TheOrangeIsItsOwn(unittest.TestCase):
    def test_a_token_a_banner_a_lamp_and_a_cap(self):
        css = (pathlib.Path(__file__).resolve().parents[1] / "static/css/cockpit.css").read_text()
        self.assertRegex(css, r"--orange:#[0-9a-f]{6}; --orange-hi:#[0-9a-f]{6};")
        rules = css_rules(css)
        for sel in (".banner.orange", ".lamp.orange", ".cap.orange", ".cf-list li.moves b"):
            with self.subTest(sel=sel):
                self.assertTrue(any(sel in s and "--orange" in decl for s, _, decl in rules), sel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
