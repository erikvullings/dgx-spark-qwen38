#!/usr/bin/env python3
"""Every text lane's chat template, as its installer serves it, takes what the real clients send.

opencode 1.19.4 sends two system messages before the user's, and the published template of the
Ornith 1.5 122B lane refused the second ("System message must be at the beginning"): every
opencode request on that lane got a 400 from v1.38.6 to v1.40.1, and nothing saw it, as each
release proved the clients on the lane that happened to serve (found on 2026-10-24).

This takes each text lane's chat template at its pinned revision on Hugging Face (chat_template.jinja,
or the chat_template of tokenizer_config.json), patches it as its installer does (patch-template.py,
with the lane's TEMPLATE_PATCHES), and renders, as transformers renders it for vLLM and SGLang, every
message shape the real clients send (tests/fixtures/client-shapes.json), with tools and without. A
template that raises fails the run, naming the lane and the client. So do a text target of
switch-model.sh this check does not know (a variant is a target: Kolibri's FP8, its default, and its
NVFP4 each have their pin) and a lane whose installer patches its template otherwise than the table
below says.

Network: the templates come from huggingface.co. Usage: lane-templates-check.py (no argument).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHAPES = REPO / "tests" / "fixtures" / "client-shapes.json"

# How each vLLM target's installer serves its template: None as published, else the environment its
# installer runs patch-template.py with. lane_specs() checks this against the installers themselves.
# A target, not a lane: the Kolibri lane's two checkpoints each have their pin, and the check read only
# the NVFP4's while the FP8 is the lane's default (found in review, 2026-10-24).
VLLM_LANES = {
    "flash-local": ("install-flash-local.sh", "FLASH_LOCAL", {"LEAN_DEFAULT": "0"}),
    "ornith-122b": ("install-ornith-122b.sh", "ORNITH_122B", {"TEMPLATE_PATCHES": "system-reminder"}),
    "kolibri": ("install-kolibri.sh", "KOLIBRI_FP8", None),
    "kolibri-nvfp4": ("install-kolibri.sh", "KOLIBRI_NVFP4", None),
    "nemotron": ("install-nemotron.sh", "NEMOTRON_BOLT", None),
    "nemotron-super": ("install-nemotron-super.sh", "NEMOTRON_SUPER", None),
}
# switch-model.sh's targets that serve no chat template
NOT_TEXT = {"image", "image-turbo", "video"}


def pin(text, prefix):
    """The (repo, revision) an installer pins under PREFIX_REPO and PREFIX_REV."""
    repo = re.search(rf'^{prefix}_REPO="([^"]+)"', text, re.M)
    rev = re.search(rf'^{prefix}_REV="([0-9a-f]{{40}})"', text, re.M)
    if not (repo and rev):
        raise SystemExit(f"{prefix}_REPO and {prefix}_REV are not where this check reads them")
    return repo.group(1), rev.group(1)


def lane_specs(repo_dir=REPO):
    """Every text target: (name, repo, revision, patch environment or None). The 27B and flash targets
    are install.sh's (its case of MODEL_REPO lines), patched with the four fixes at every install and
    switch; the vLLM targets are VLLM_LANES, checked against what each installer does. Together they
    are every target of switch-model.sh but NOT_TEXT, or this stops."""
    install = (repo_dir / "install.sh").read_text()
    targets = re.findall(r'^\s+([a-z0-9-]+)\)\s+MODEL_REPO="\$([A-Z0-9_]+)_REPO"', install, re.M)
    if not targets:
        raise SystemExit("install.sh no longer maps its targets the way this check reads them")
    specs = [(t, *pin(install, p), {}) for t, p in targets]
    installers = sorted(f.name for f in repo_dir.glob("install-*.sh") if "vlane_put_unit" in f.read_text())
    known = sorted({f for f, _, _ in VLLM_LANES.values()})
    if installers != known:
        raise SystemExit(f"the vLLM lanes are {installers}, this check knows {known}: add the new lane to VLLM_LANES")
    for name, (installer, prefix, env) in VLLM_LANES.items():
        text = (repo_dir / installer).read_text()
        patched = "patch-template.py" in text
        asked = re.search(r"TEMPLATE_PATCHES=(\S+) python3", text)
        if patched != (env is not None) or (asked.group(1) if asked else None) != (env or {}).get("TEMPLATE_PATCHES"):
            raise SystemExit(f"{installer} serves its template otherwise than VLLM_LANES says for {name}")
        specs.append((name, *pin(text, prefix), env))
    switch = re.search(r'^case "\$CHOICE" in ([a-z0-9|-]+)\) ;;', (repo_dir / "switch-model.sh").read_text(), re.M)
    if not switch:
        raise SystemExit("switch-model.sh no longer lists its targets the way this check reads them")
    served, checked = set(switch.group(1).split("|")) - NOT_TEXT, {name for name, _, _, _ in specs}
    if served != checked:
        raise SystemExit(f"switch-model.sh serves the text targets {sorted(served - checked)} this check does not"
                         f" know, and it knows {sorted(checked - served)} the switch does not serve: add them")
    return specs


def fetch(repo, rev):
    """The chat template the engine reads at that revision."""
    def get(path):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    return r.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if attempt == 2:
                    raise
            except OSError:
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)
    tpl = get("chat_template.jinja")
    if tpl is not None:
        return tpl
    cfg = get("tokenizer_config.json")
    t = json.loads(cfg or "{}").get("chat_template")
    if isinstance(t, list):          # named templates: the engine takes "default"
        t = next((x["template"] for x in t if x.get("name") == "default"), None)
    if not isinstance(t, str):
        raise SystemExit(f"{repo}@{rev[:12]} has no chat template where the engines look")
    return t


def served(repo, rev, template, env):
    """The template as the lane's installer serves it: patch-template.py on a cache of one snapshot."""
    if env is None:
        return template
    with tempfile.TemporaryDirectory(prefix="lane-tpl-") as d:
        snap = Path(d, "hub", f"models--{repo.replace('/', '--')}", "snapshots", rev)
        snap.mkdir(parents=True)
        (snap / "chat_template.jinja").write_text(template)
        out = Path(d, "served.jinja")
        r = subprocess.run([sys.executable, str(REPO / "patch-template.py"), d, str(out), rev, repo],
                           capture_output=True, text=True, env={**os.environ, **env})
        if r.returncode != 0:
            raise SystemExit(f"patch-template.py refused {repo}@{rev[:12]}:\n{r.stdout}{r.stderr}")
        return out.read_text()


def environment():
    """A Jinja environment as transformers compiles chat templates with: sandboxed, blocks trimmed,
    loop controls, raise_exception, tojson and strftime_now."""
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


TOOLS = [{"type": "function", "function": {
    "name": "read", "description": "Read a file of the project.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def messages(roles):
    """A conversation of these roles, with a tool call where the shape has one."""
    out = []
    for i, role in enumerate(roles):
        if role == "assistant_tool_call":
            out.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "read", "arguments": {"path": "hello.txt"}}}]})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": f"call_{i - 1}", "content": "7351"})
        else:
            out.append({"role": role, "content": f"A {role} message, number {i}."})
    return out


def render(env, template, roles, tools):
    """None when the template renders this conversation, else why it does not."""
    try:
        env.from_string(template).render(messages=messages(roles), tools=tools, add_generation_prompt=True,
                                         bos_token="", eos_token="")
        return None
    except Exception as e:  # noqa: BLE001: any refusal of the template is the finding
        return f"{type(e).__name__}: {e}"


# A template that refuses a system message past the first one, as Ornith's published one did: the
# renderer must see its refusal, or this check would pass on anything.
CANARY = ("{%- for m in messages %}{%- if m.role == 'system' and not loop.first %}"
          "{{- raise_exception('System message must be at the beginning.') }}{%- endif %}{%- endfor %}")


def main():
    shapes = json.loads(SHAPES.read_text())["shapes"]
    env = environment()
    if render(env, CANARY, ["system", "system", "user"], None) is None:
        raise SystemExit("the renderer did not see the canary's refusal: this check would pass on anything")
    failed = 0
    for name, repo, rev, patch in lane_specs():
        template = served(repo, rev, fetch(repo, rev), patch)
        how = "as published" if patch is None else f"patched ({patch.get('TEMPLATE_PATCHES', 'all four fixes')})"
        print(f"{name}: {repo}@{rev[:12]}, {how}")
        for shape in shapes:
            for tools in (None, TOOLS):
                why = render(env, template, shape["roles"], tools)
                if why:
                    failed += 1
                    print(f"  REFUSED {shape['client']}{' with tools' if tools else ''}: {why}")
    print(f"{failed} refusal(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
