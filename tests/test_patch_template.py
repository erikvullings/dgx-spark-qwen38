#!/usr/bin/env python3
"""patch-template.py on a fixture built from the script's own anchor
constants: every patch applied, idempotent message on a pre-patched template,
exact-revision snapshot selection, custom repo argument, and a Claude Code
tool_reference block rendered instead of raising. No network."""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import shutil

import jinja2

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_DIR, "patch-template.py")

spec = importlib.util.spec_from_file_location("pt", SCRIPT)
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)


def make_fixture(base: str, repo: str, sha: str, body: str) -> str:
    d = f"{base}/hub/models--{repo.replace('/', '--')}/snapshots/{sha}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/chat_template.jinja", "w") as f:
        f.write(body)
    return f"{d}/chat_template.jinja"


def run(base: str, out: str, rev: str, repo: str) -> str:
    r = subprocess.run([sys.executable, SCRIPT, base, out, rev, repo],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def main() -> None:
    base = tempfile.mkdtemp(prefix="tpl-fixture-")
    try:
        env = {k: v for k, v in os.environ.items() if k not in ("TEMPLATE_PATCHES", "LEAN_DEFAULT")}
        sha = "c" * 40
        repo = "org/custom-model"
        # Every anchor the script looks for, in the order the upstream template has
        # them. A fixture missing one makes the patcher refuse the whole template,
        # which is the behaviour tested below for a REAL upstream change and not
        # something this fixture should be triggering by accident.
        stock_body = (
            "HEAD\n"
            "{%- macro render_content(content, do_vision_count, is_system_content=false) %}\n"
            "    {%- if content is string %}\n"
            "        {{- content }}\n"
            "    {%- elif content is iterable and content is not mapping %}\n"
            "        {%- for item in content %}\n"
            "            {%- if 'image' in item %}\n"
            "                {{- '<image>' }}\n"
            + pt.TOOL_REF_ANCHOR
            + "            {%- endif %}\n"
            + "        {%- endfor %}\n"
            "    {%- endif %}\n"
            "{%- endmacro %}\n"
            + pt.EFFORT_ANCHOR + "\n" + pt.MSG_ANCHOR + "\nMID\n"
            + pt.LEAN_ANCHOR + pt.SYSTEM_ANCHOR + "\nTAIL\n"
        )
        make_fixture(base, repo, sha, stock_body)
        out = os.path.join(base, "out.jinja")
        log = run(base, out, sha, repo)
        assert log.count("applied") == 5, log
        patched = open(out).read()
        assert "'minimal'" in patched, "effort patch must map the minimal tier"
        assert "resolved_reasoning_effort == 'lean'" in patched, "lean patch must add the level"
        assert "default('lean')" in patched, "lean must become the default"
        assert "lean (default), xhigh, medium, and low" in patched, "the refusal must name lean"
        assert "<system-reminder>" in patched, "system patch must render reminders"
        assert pt.TOOL_REF_MARKER in patched, "tool-reference patch must render the reference"
        assert patched.startswith("HEAD\n") and patched.endswith("TAIL\n")
        # A Claude Code ToolSearch result, as SGLang's Anthropic endpoint hands it to the
        # template: the stock loop's last branch raised "Unexpected item type in content"
        # and every turn of the session was a 500 (2026-10-25). The patched loop names the
        # tool; the schemas themselves come in the next request's tools.
        ref_repo, ref_sha = "org/tool-search", "f" * 40
        ref_body = (
            "{%- macro render_content(content, do_vision_count, is_system_content=false) %}\n"
            "    {%- if content is string %}\n"
            "        {{- content }}\n"
            "    {%- elif content is iterable and content is not mapping %}\n"
            "        {%- for item in content %}\n"
            "            {%- if 'image' in item %}\n"
            "                {{- '<image>' }}\n"
            + pt.TOOL_REF_ANCHOR
            + "            {%- endif %}\n"
            + "        {%- endfor %}\n"
            "    {%- endif %}\n"
            "{%- endmacro %}\n"
            "{{- render_content(messages[0].content, false) }}"
        )
        make_fixture(base, ref_repo, ref_sha, ref_body)
        ref_out = os.path.join(base, "tool-reference.jinja")
        r = subprocess.run([sys.executable, SCRIPT, "--patches=tool-reference", base, ref_out, ref_sha, ref_repo],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "patch 'tool-reference': applied" in r.stdout, r.stdout
        rendered = jinja2.Template(open(ref_out).read()).render(
            messages=[{"content": [{"type": "tool_reference", "name": "mcp__chrome-devtools__click"}]}])
        assert "[tool available: mcp__chrome-devtools__click]" in rendered, rendered
        # The same template again leaves the file as it is, mtime included: the engine
        # reads it at start, and install.sh keeps a running engine only when nothing it
        # reads was written after it started (every run used to rewrite it, 2026-09-23).
        os.utime(out, (1_000_000_000, 1_000_000_000))
        log_same = run(base, out, sha, repo)
        assert "patched template unchanged" in log_same, log_same
        assert os.stat(out).st_mtime == 1_000_000_000, "an identical template was rewritten"
        assert open(out).read() == patched
        # A template that already carries all five fixes: succeed, change nothing.
        make_fixture(base, repo, sha, patched)
        log2 = run(base, out, sha, repo)
        assert log2.count("already present") == 5, log2
        assert open(out).read() == patched
        # Exact-revision selection with a newer decoy snapshot.
        decoy = make_fixture(base, repo, "d" * 40, "DECOY " + stock_body)
        os.utime(decoy, None)
        run(base, out, sha, repo)
        assert not open(out).read().startswith("DECOY"), "pinned snapshot must win"
        # A lane whose template has no reasoning effort of its own asks for one fix by name
        # (Ornith 1.5 122B, 2026-10-24, whose template refused a system message past the first one,
        # as opencode 1.19.4 sends): that fix alone, no effort line, every other byte kept. Without
        # the name the patcher refuses that template, as it refuses an upstream change.
        orn_repo, orn_sha = "org/no-effort", "e" * 40
        orn_body = "HEAD\n" + pt.SYSTEM_ANCHOR + "\nTAIL\n"
        make_fixture(base, orn_repo, orn_sha, orn_body)
        orn_out = os.path.join(base, "orn.jinja")
        argv = [sys.executable, SCRIPT, base, orn_out, orn_sha, orn_repo]
        r = subprocess.run(argv[:2] + ["--patches=system-reminder"] + argv[2:], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "patch 'system-reminder': applied" in r.stdout, r.stdout
        assert "reasoning effort" not in r.stdout and r.stdout.count("patch '") == 1, r.stdout
        assert open(orn_out).read() == orn_body.replace(pt.SYSTEM_ANCHOR, pt.SYSTEM_PATCHED)
        os.remove(orn_out)
        r = subprocess.run(argv, capture_output=True, text=True, env=env)
        assert r.returncode != 0 and "anchor not found" in r.stdout + r.stderr, r.stdout + r.stderr
        assert not os.path.exists(orn_out), "a refused template was written"
        r = subprocess.run(argv[:2] + ["--patches=system-reminders"] + argv[2:], capture_output=True, text=True, env=env)
        assert r.returncode != 0 and "the fixes are" in r.stdout + r.stderr, r.stdout + r.stderr
        # A selection left in the shell reaches no caller: the 27B's template keeps its five fixes with
        # TEMPLATE_PATCHES=system-reminder exported, which made ./install.sh drop its effort fixes when
        # v1.40.2 as first pushed read it from the environment (found in review, 2026-10-24).
        ambient = os.path.join(base, "ambient.jinja")
        r = subprocess.run([sys.executable, SCRIPT, base, ambient, sha, repo], capture_output=True, text=True,
                           env={**env, "TEMPLATE_PATCHES": "system-reminder"})
        assert r.returncode == 0, r.stdout + r.stderr
        assert r.stdout.count("patch '") == 5 and "reasoning_effort" in open(ambient).read(), r.stdout
        # Ornith's installer asks for the tool-reference fix too (2026-10-26): its render_content loop is the
        # Qwen one, and a Claude Code ToolSearch result was a 400 "Unexpected item type in content." on the
        # FP4-Local lane, as it was a 500 on SGLang. The installer's own TEMPLATE_FIXES line, on a template
        # with both anchors, must patch the reference loop, not the system message alone.
        fixes = re.search(r'^TEMPLATE_FIXES=\((--patches=[^)]*)\)$',
                          open(os.path.join(REPO_DIR, "install-ornith-122b.sh")).read(), re.M)
        assert fixes, "install-ornith-122b.sh sets no --patches TEMPLATE_FIXES"
        both_repo, both_sha = "org/ornith-like", "c" * 40
        make_fixture(base, both_repo, both_sha, "HEAD\n" + pt.SYSTEM_ANCHOR + "\n" + ref_body + "\nTAIL\n")
        both_out = os.path.join(base, "ornith-like.jinja")
        r = subprocess.run([sys.executable, SCRIPT, fixes.group(1), base, both_out, both_sha, both_repo],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        both = open(both_out).read()
        assert pt.SYSTEM_PATCHED in both, "Ornith's fixes lost the system-reminder"
        assert pt.TOOL_REF_PATCHED in both, f"Ornith's fixes ({fixes.group(1)}) leave the tool_reference loop raising"
        # A template that raises on an unknown item somewhere else than the loop the fix knows (a renamed
        # macro, an inline loop, as a next checkpoint may have) stops the install, as an upstream change
        # does for every other fix: the guard looked at the macro's name and let it through, every turn
        # after a ToolSearch a 500 again (code review, 2026-10-26). One that raises on none needs no fix.
        # Its words are no guard either: a raise worded otherwise went through as well (code review, 2026-10-26).
        for n, (body, ok) in enumerate((
                ("{%- for item in content %}{%- if item.type == 'text' %}{{ item.text }}{%- else %}"
                 "{{ raise_exception('Unexpected item type in content.') }}{%- endif %}{%- endfor %}", False),
                ("{%- for item in content %}{%- if item.type == 'text' %}{{ item.text }}{%- else %}"
                 "{{ raise_exception('Unexpected content item type: ' ~ item.type) }}{%- endif %}{%- endfor %}", False),
                ("{%- for item in content %}{%- if item.type == 'text' %}{{ item.text }}{%- else %}"
                 "{{ raise_exception ('Unknown item') }}{%- endif %}{%- endfor %}", False),
                ("{%- set fail = raise_exception %}{%- for item in content %}{%- if item.type == 'text' %}"
                 "{{ item.text }}{%- else %}{{ fail('Unknown item') }}{%- endif %}{%- endfor %}", False),
                ("{%- for item in content %}{{ item.text }}{%- endfor %}", True))):
            other_repo, other_sha = "org/next-checkpoint", "abcdf"[n] * 40
            make_fixture(base, other_repo, other_sha, body)
            other_out = os.path.join(base, f"next-{n}.jinja")
            r = subprocess.run([sys.executable, SCRIPT, "--patches=tool-reference", base, other_out, other_sha,
                                other_repo], capture_output=True, text=True, env=env)
            if ok:
                assert r.returncode == 0 and "patch 'tool-reference': not needed" in r.stdout, r.stdout + r.stderr
            else:
                assert r.returncode != 0, f"a loop that raises was let through:\n{r.stdout}{r.stderr}"
                assert "anchor not found" in r.stdout + r.stderr, r.stdout + r.stderr
                assert not os.path.exists(other_out), "a template the fix could not reach was written"
        print("test_patch_template: OK")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
