#!/usr/bin/env python3
"""A run of ./install.sh that changes nothing restarts nothing.

v1.18.5 kept the engine when nothing it reads changed, and everything around it went on
restarting at every run: the proxy (every request in flight through it cut), opencode-web
up to three times (the Agent tab's turn ended) and the cockpit twice. On the reference box
on 2026-09-23: 49 starts of opencode-web and 36 of the cockpit. And on a 1m box step 7
wrote the table's bounds over the pair the end-of-install fit had set, which the fit then
wrote back, a backup each time: 79 in ~/.config/opencode by 2026-09-24 (found in review).

Each service is now restarted when it is not running, when this run changed what it runs,
or when it started before the last change of a file it reads; and those files are written
only when their content changes, so their date means something.
"""
import json
import os
import pathlib
import shutil
import re
import subprocess
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
INSTALL = (REPO / "install.sh").read_text()
# install.sh's is lanes.sh's since v1.30, which it sources (the lanes' installers share it)
SCRIPTS = {name: (REPO / name).read_text() for name in
           ("lanes.sh", "dashboard/install-dashboard.sh", "dashboard/install-agent.sh")}


def helper(text):
    if text is SCRIPTS["lanes.sh"]:
        return text + "\n"                  # the library whole, as install.sh sources it
    start = text.index("stale_since(){")
    return text[start:text.index("\n}\n", start) + 3]


def stale(text, active="active", started=None, files=()):
    d = pathlib.Path(tempfile.mkdtemp(prefix="stale-"))
    (d / "systemctl").write_text(
        "#!/bin/sh\ncase \"$*\" in *ActiveState*) echo \"$FAKE_ACTIVE\";; "
        "*ExecMainStartTimestamp*) [ -n \"$FAKE_STARTED\" ] && echo \"@$FAKE_STARTED\";; esac\n")
    (d / "systemctl").chmod(0o755)
    paths = []
    for i, age in enumerate(files):
        f = d / f"f{i}"
        f.write_text("x")
        os.utime(f, (time.time() - age, time.time() - age))
        paths.append(str(f))
    script = helper(text) + "if stale_since unit.service " + " ".join(paths) + "; then echo RESTART; else echo KEEP; fi\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": f"{d}:/usr/bin:/bin", "FAKE_ACTIVE": active,
                            "FAKE_STARTED": "" if started is None else str(int(started))})
    return r.stdout.strip()


class StaleSince(unittest.TestCase):
    def test_every_copy_says_the_same(self):
        now = time.time()
        for name, text in SCRIPTS.items():
            with self.subTest(name):
                self.assertEqual(stale(text, active="inactive"), "RESTART")
                self.assertEqual(stale(text, started=now - 100, files=(500, 300)), "KEEP")
                self.assertEqual(stale(text, started=now - 100, files=(500, 10)), "RESTART")
                self.assertEqual(stale(text, started=None, files=(500,)), "RESTART")

    def test_install_sh_has_the_librarys_before_it_asks(self):
        self.assertNotIn("stale_since(){", INSTALL, "one copy: lanes.sh's")
        self.assertLess(INSTALL.index('. "$REPO_DIR/lanes.sh"'), INSTALL.index('stale_since "$KEEPALIVE_UNIT"'))


class EachRestartHasAReason(unittest.TestCase):
    def test_opencode_web_after_step_7_only_when_a_config_changed(self):
        i = INSTALL.index('sudo systemctl restart opencode-web.service \\\n      && echo "opencode-web.service restarted')
        guard = INSTALL[INSTALL.rfind("\n  if ", 0, i):i]
        self.assertIn('"$(oc_configs_sum)" != "$OC_SUM_BEFORE"', guard)
        self.assertLess(INSTALL.index('OC_SUM_BEFORE="$(oc_configs_sum)"'), INSTALL.index("python3 - <<'PYEOF' || die \"could not write the opencode provider config\""))

    def test_the_proxy_only_for_a_new_engine_new_code_or_a_new_unit(self):
        for m in re.finditer(r'sudo systemctl restart "\$KEEPALIVE_UNIT"', INSTALL):
            guard = INSTALL[INSTALL.rfind("if ", 0, m.start()):m.start()]
            self.assertIn("stale_since \"$KEEPALIVE_UNIT\"", guard, INSTALL[m.start() - 200:m.start()])
            self.assertIn('"$KA_CHANGED" -eq 1', guard)
        for write in ('install -m 755 "$REPO_DIR/keepalive-proxy.py" "$CONFIG_DIR/keepalive-proxy.py"; KA_CHANGED=1',
                      'sudo install -m 644 "$TMP_KA" "/etc/systemd/system/$KEEPALIVE_UNIT"; KA_CHANGED=1'):
            self.assertIn(write, INSTALL)

    def test_the_cockpit_and_opencode_web_installers(self):
        for name, marker in (("dashboard/install-dashboard.sh", 'DASH_CHANGED'),
                             ("dashboard/install-agent.sh", 'AGENT_CHANGED')):
            text = SCRIPTS[name]
            i = text.index('sudo systemctl try-restart "$UNIT"')
            guard = text[text.rfind("\nif ", 0, i):i]
            self.assertIn(f'"${marker}" -eq 1', guard, name)
            self.assertIn("stale_since", guard, name)


class ABoxBootingAnotherLaneGetsTheAgentTab(unittest.TestCase):
    """install-agent.sh restarts opencode-web on a newer opencode and points it at the one on
    the PATH, and it ran at step 10 only: a box whose boot lane is not a text one takes another
    branch, which reinstalled the cockpit alone, and it kept the opencode it started with until
    its next boot (the reference box on the FP4-Local lane, 2026-10-22, where a new pin's reason
    was the Bun under opencode). That branch runs it now, after the cockpit, as step 10 does."""
    GUARD = '[ "$OPENCODE" -eq 1 ] && command -v opencode >/dev/null 2>&1; then\n'
    AGENT = '"$REPO_DIR/dashboard/install-agent.sh" \\\n'
    TEXT = SCRIPTS["dashboard/install-agent.sh"]

    def test_its_branch_runs_the_agent_installer_after_the_cockpit_as_step_10_does(self):
        start = INSTALL.index("# Updated, not switched to.")
        seg = INSTALL[start:INSTALL.index("# --no-smoke always", start)]
        self.assertIn('if "$REPO_DIR/dashboard/install-dashboard.sh"; then', seg,
                      "the Agent tab only once the cockpit that relays it is installed")
        self.assertLess(seg.index('"$REPO_DIR/dashboard/install-dashboard.sh"'), seg.index(self.AGENT))
        self.assertIn(self.GUARD + "        " + self.AGENT, seg)
        self.assertIn('|| echo "NOTE: the Agent view did not install', seg, "a failure is a note, as at step 10")
        step10 = INSTALL[INSTALL.index('step "10/10 Spark Cockpit'):]
        self.assertIn(self.GUARD + "          " + self.AGENT, step10, "the same condition as step 10's")

    def restarted(self, binary_after_start, served="1.19.4", on_disk="1.19.4"):
        """The script's own guard, run as written: the opencode-web unit, the opencode binary (its
        date after or before the server's start), the version the server answers with (None: it
        does not answer) and the one `opencode --version` gives."""
        t = pathlib.Path(tempfile.mkdtemp(prefix="agent-bin-"))
        self.addCleanup(shutil.rmtree, t, True)
        bin_, home = t / "bin", t / "home"
        bin_.mkdir(); (home / ".config" / "qwen38").mkdir(parents=True); (home / ".config" / "opencode").mkdir()
        start = 1_800_000_000
        oc = t / "opencode"
        for f in (oc, t / "opencode-web.env", home / ".config/qwen38/opencode.json", home / ".config/opencode/opencode.json"):
            f.write_text("x\n")
            os.utime(f, (start - 100, start - 100))
        if binary_after_start:
            os.utime(oc, (start + 100, start + 100))
        log = t / "calls"
        (bin_ / "systemctl").write_text(
            f'#!/bin/sh\ncase "$*" in *ActiveState*) echo active ;; *ExecMainStartTimestamp*) echo @{start} ;; esac\n')
        (bin_ / "sudo").write_text(f'#!/bin/sh\necho "$*" >> "{log}"\n')
        for f in ("systemctl", "sudo"):
            (bin_ / f).chmod(0o755)
        func = self.TEXT[self.TEXT.index("stale_since(){"):self.TEXT.index("\n}\n", self.TEXT.index("stale_since(){")) + 3]
        guard = self.TEXT[self.TEXT.index('SERVED_VERSION="$(health'):]
        guard = guard[:guard.index("\nfi\n") + 4]
        answer = "" if served is None else '{"healthy":true,"version":"%s"}' % served
        script = (f'set -euo pipefail\nUNIT=opencode-web.service; AGENT_CHANGED=0\n'
                  f'OPENCODE_BIN="{oc}"; ENV_FILE="{t}/opencode-web.env"; OC_VERSION="{on_disk}"\n'
                  f"health(){{ printf '%s' '{answer}'; }}\n" + func + "\n" + guard)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(home)})
        self.assertEqual(r.returncode, 0, r.stderr)
        return log.exists() and "try-restart opencode-web.service" in log.read_text()

    def test_a_newer_opencode_restarts_opencode_web(self):
        self.assertTrue(self.restarted(binary_after_start=True))

    def test_the_same_opencode_does_not(self):
        self.assertFalse(self.restarted(binary_after_start=False))
        self.assertFalse(self.restarted(binary_after_start=False, served="v1.19.4"), "a v in front is the same")

    def test_an_opencode_updated_with_an_older_date_restarts_it_by_its_version(self):
        """npm and Homebrew keep the date their archive gives the binary: a server started after
        that date ran the old opencode on, its binary older on disk than its start (found in
        review, 2026-10-22). The version it answers with tells."""
        self.assertTrue(self.restarted(binary_after_start=False, served="1.18.35", on_disk="1.19.4"))

    def test_a_server_that_does_not_answer_is_judged_by_dates_as_before(self):
        self.assertFalse(self.restarted(binary_after_start=False, served=None))
        self.assertTrue(self.restarted(binary_after_start=True, served=None))


class TheCockpitFollowsTheReleaseItNames(unittest.TestCase):
    """The cockpit reads the release it names (CHANGELOG.md's first heading) at start, and
    install-dashboard.sh restarted it only for its python or its unit: a box updated to
    v1.20.0, whose update touched only the page's static files and the CHANGELOG, kept
    naming v1.19.0 (2026-10-01). The guard, run as written against a fake systemctl."""
    TEXT = SCRIPTS["dashboard/install-dashboard.sh"]
    FUNC = TEXT[TEXT.index("stale_since(){"):TEXT.index("\n}\n", TEXT.index("stale_since(){")) + 3]
    GUARD = TEXT[TEXT.index('if [ "$DASH_CHANGED" -eq 1 ] || stale_since'):]
    GUARD = GUARD[:GUARD.index("\nfi\n") + 4]

    def restarted(self, changelog_after_start):
        t = pathlib.Path(tempfile.mkdtemp(prefix="dash-release-"))
        here, repo, bin_ = t / "dashboard", t, t / "bin"
        here.mkdir(); bin_.mkdir()
        start = 1_800_000_000
        for f in (here / "cockpit.py", repo / "oc-fit-limits.py", repo / "CHANGELOG.md"):
            f.write_text("x\n")
            os.utime(f, (start - 100, start - 100))
        if changelog_after_start:
            os.utime(repo / "CHANGELOG.md", (start + 100, start + 100))
        log = t / "calls"
        (bin_ / "systemctl").write_text(
            f'#!/bin/sh\ncase "$*" in *ActiveState*) echo active ;; *ExecMainStartTimestamp*) echo @{start} ;; esac\n')
        (bin_ / "sudo").write_text(f'#!/bin/sh\necho "$*" >> "{log}"\n')
        for f in ("systemctl", "sudo"):
            (bin_ / f).chmod(0o755)
        script = (f'set -euo pipefail\nUNIT=qwen38-dashboard.service; DASH_CHANGED=0\n'
                  f'HERE="{here}"; REPO_DIR="{repo}"\n' + self.FUNC + "\n" + self.GUARD)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={"PATH": f"{bin_}:/usr/bin:/bin"})
        self.assertEqual(r.returncode, 0, r.stderr)
        return log.exists() and "try-restart qwen38-dashboard.service" in log.read_text()

    def test_a_new_release_restarts_it(self):
        self.assertTrue(self.restarted(changelog_after_start=True))

    def test_the_same_release_does_not(self):
        self.assertFalse(self.restarted(changelog_after_start=False))


class TheGeneratedConfigKeepsItsFit(unittest.TestCase):
    """The generator of ~/.config/qwen38/opencode.json, run twice in one config dir with the
    environment install.sh gives it: once as a fresh install, then again after the
    end-of-install fit wrote the pool's pair into the file."""

    BODY = None

    def run_generator(self, d, mode):
        start = INSTALL.index("OC_CONFIG_DIR=\"$CONFIG_DIR\" python3 - <<'PYEOF'")
        body = INSTALL[INSTALL.index("\n", start) + 1:INSTALL.index("\nPYEOF\n", start)]
        ctx, out = ("700000", "200000") if mode == "1m" else ("173000", "64000")
        env = dict(os.environ, OC_LANE="27b", OC_27B="1", OC_FLASH="0", OC_PORT="30001", OC_CTX=ctx,
                   OC_OUT=out, OC_LABEL="local, 1M", OC_CONTEXT_MODE=mode, OC_27B_CTX=ctx, OC_27B_OUT=out,
                   OC_KEEP="38000", OC_PIN="1", OC_CONFIG_DIR=str(d))
        r = subprocess.run(["python3", "-c", body], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        f = d / "opencode.json"
        lim = json.loads(f.read_text())["provider"]["qwen38"]["models"]["qwen3.8-27b"]["limit"]
        return (lim["context"], lim["output"]), os.stat(f).st_mtime, f.read_text()

    def fitted_dir(self, ctx, out):
        d = pathlib.Path(tempfile.mkdtemp(prefix="oc-gen-"))
        self.run_generator(d, "1m")
        f = d / "opencode.json"
        f.write_text(f.read_text().replace('"context": 700000', f'"context": {ctx}').replace(
            '"input": 700000', f'"input": {ctx}').replace('"output": 200000', f'"output": {out}'))
        os.utime(f, (1000, 1000))
        return d

    def test_a_fit_under_the_1m_bounds_is_kept(self):
        pair, _, _ = self.run_generator(self.fitted_dir(559000, 186000), "1m")
        self.assertEqual(pair, (559000, 186000))

    def test_the_same_config_is_not_rewritten(self):
        d = self.fitted_dir(559000, 186000)
        before = (d / "opencode.json").read_text()
        _, mtime, text = self.run_generator(d, "1m")
        self.assertEqual(text, before)
        self.assertEqual(mtime, 1000, "an identical config was written again, and its date moved")

    def keep_after(self, d, mode="1m", lane="27b"):
        """OC_KEEP as install.sh resolves it for the config in d."""
        start = INSTALL.index("# On 1m the table gives bounds and the fit to the pool sets the pair")
        block = INSTALL[start:INSTALL.index("\nfi\n", INSTALL.index("  if [ -n \"$OC_FIT_CTX\" ]; then", start)) + 4]
        script = (f'set -euo pipefail\ndie(){{ echo "DIE $*"; exit 1; }}\nREPO_DIR="{REPO}"\nCONFIG_DIR="{d}"\n'
                  f'CONTEXT_MODE={mode}\nLANE={lane}\nOC_CTX=700000\nOC_OUT=200000\nOC_KEEP=170000\n'
                  + block + 'echo "KEEP=$OC_KEEP"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return int(r.stdout.strip().rsplit("KEEP=", 1)[1])

    def test_the_compaction_goes_with_the_fitted_pair(self):
        """Sized from the table's bound, the compaction was set back at every run and the fit
        at the end set it again: two writes, two backups and two opencode-web restarts per
        update on the reference box (2026-09-25). A fitted pair keeps its own."""
        want = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", "560000"],
                                  capture_output=True, text=True).stdout)
        self.assertEqual(self.keep_after(self.fitted_dir(560000, 186000)), want)
        self.assertLess(want, 170000)
        # the table's own pair, no fit, and the other lanes keep the table's value
        self.assertEqual(self.keep_after(self.fitted_dir(700000, 200000)), 170000)
        self.assertEqual(self.keep_after(self.fitted_dir(560000, 186000), mode="native"), 170000)
        self.assertEqual(self.keep_after(self.fitted_dir(560000, 186000), lane="flash"), 170000)
        self.assertEqual(self.keep_after(pathlib.Path(tempfile.mkdtemp(prefix="oc-none-"))), 170000)

    def test_a_native_install_writes_the_table(self):
        pair, _, _ = self.run_generator(self.fitted_dir(559000, 186000), "native")
        self.assertEqual(pair, (173000, 64000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
