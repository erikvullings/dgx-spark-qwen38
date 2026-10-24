#!/usr/bin/env python3
"""Every text lane's chat template, as its installer serves it, takes what the real clients send.

opencode 1.19.4 sends two system messages before the user's, and the published template of the
Ornith 1.5 122B lane refused the second ("System message must be at the beginning"): every
opencode request on that lane got a 400 from v1.38.6 to v1.40.1, and nothing saw it, as each
release proved the clients on the lane that happened to serve (found on 2026-10-24).

This takes each text lane's chat template at its pinned revision on Hugging Face (chat_template.jinja,
or the chat_template of tokenizer_config.json, the one named tool_use when tools are sent and it has
one, as transformers picks), patches it as its installer does (patch-template.py with the lane's
--patches, from chat_template.jinja only, as the installer reads it), and renders, as transformers
renders it for vLLM and SGLang, every message shape the real clients send
(tests/fixtures/client-shapes.json), with tools and without, each message's content as a string and as
a list of parts (what vLLM hands a template it reads as taking OpenAI's content format). A template
that raises fails the run, naming the lane and the client. So do a text target of
switch-model.sh this check does not know (a variant is a target: Kolibri's FP8, its default, and its
NVFP4 each have their pin) and a lane whose installer patches its template otherwise than the table
below says.

Network: the templates come from huggingface.co. Usage: lane-templates-check.py (no argument).
"""
import http.client
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

# How each vLLM target's installer serves its template: None as published, else how its installer
# runs patch-template.py ("patches": its --patches option, the rest its environment). lane_specs()
# checks this against the installers themselves.
# A target, not a lane: the Kolibri lane's two checkpoints each have their pin, and the check read only
# the NVFP4's while the FP8 is the lane's default (found in review, 2026-10-24).
VLLM_LANES = {
    "flash-local": ("install-flash-local.sh", "FLASH_LOCAL", {"LEAN_DEFAULT": "0"}),
    "ornith-122b": ("install-ornith-122b.sh", "ORNITH_122B", {"patches": "system-reminder"}),
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
        asked = re.search(r'patch-template\.py" --patches=(\S+) ', text)
        if patched != (env is not None) or (asked.group(1) if asked else None) != (env or {}).get("patches"):
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


# Hugging Face answers a runner's anonymous requests from shared addresses: a 429 or a brief 5xx
# is waited out (its Retry-After when it gives one), not taken for a broken template (2026-10-24).
RETRY = (408, 425, 429, 500, 502, 503, 504)


def fetch(repo, rev, tries=5, sleep=time.sleep):
    """(the chat template the engine reads at that revision, the file it comes from): chat_template.jinja,
    else the chat_template of tokenizer_config.json, a dict of them by name when it names several."""
    def get(path):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
        for attempt in range(tries):
            wait = 2 ** (attempt + 1)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    return r.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code not in RETRY or attempt == tries - 1:
                    raise
                after = (e.headers or {}).get("Retry-After") or ""
                wait = min(int(after), 60) if after.isdigit() else wait
            except (OSError, http.client.HTTPException):    # a reset, a timeout, a body cut short
                if attempt == tries - 1:
                    raise
            sleep(wait)
    tpl = get("chat_template.jinja")
    if tpl is not None:
        return tpl, "chat_template.jinja"
    t = json.loads(get("tokenizer_config.json") or "{}").get("chat_template")
    if isinstance(t, list):
        t = {x.get("name"): x.get("template") for x in t if isinstance(x, dict) and x.get("template")}
    if not t:
        raise SystemExit(f"{repo}@{rev[:12]} has no chat template where the engines look")
    return t, "tokenizer_config.json"


def pick(templates, tools):
    """The template transformers renders: the one named tool_use when tools are sent and it has one,
    else the one named default; a single template is itself."""
    if isinstance(templates, str):
        return templates
    name = "tool_use" if tools and "tool_use" in templates else "default"
    if name not in templates:
        raise SystemExit(f"named chat templates without a {name!r} one: {sorted(templates)}")
    return templates[name]


def served(repo, rev, fetched, spec):
    """The template as the lane's installer serves it: as published, or patch-template.py on a cache of
    one snapshot, which reads chat_template.jinja only: a patched lane without one would pass here and
    stop its installer (found in review, 2026-10-24)."""
    template, source = fetched
    if spec is None:
        return template
    if source != "chat_template.jinja":
        raise SystemExit(f"{repo}@{rev[:12]}: its installer patches chat_template.jinja, which it does not have")
    spec = dict(spec)
    patches = spec.pop("patches", None)
    with tempfile.TemporaryDirectory(prefix="lane-tpl-") as d:
        snap = Path(d, "hub", f"models--{repo.replace('/', '--')}", "snapshots", rev)
        snap.mkdir(parents=True)
        (snap / "chat_template.jinja").write_text(template)
        out = Path(d, "served.jinja")
        argv = [sys.executable, str(REPO / "patch-template.py")] + ([f"--patches={patches}"] if patches else [])
        r = subprocess.run(argv + [d, str(out), rev, repo], capture_output=True, text=True, env={**os.environ, **spec})
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


# How each engine hands a template a message's content (read in their sources on 2026-10-24). vLLM
# (vllm/renderers/hf.py, _detect_content_format) reads the template: one that loops over a message's
# content takes "openai" content, a list of parts, into which vLLM turns a string too; any other
# takes "string". SGLang (sglang/srt/parser/jinja_template_utils.py,
# detect_jinja_template_content_format) reads it more simply, and leaves a string a string: a list
# reaches the template only when the client sent one and the template takes "openai". So a vLLM lane
# is rendered with the content vLLM makes, and an SGLang lane with strings, and with lists too when
# its template takes them. The detectors below are those two functions, ported (Apache-2.0).
def _var(node, name):
    import jinja2.nodes as n
    if isinstance(node, n.Name):
        return node.ctx == "load" and node.name == name
    if isinstance(node, n.Getattr) and "." in name:
        base, _, attr = name.rpartition(".")
        return node.attr == attr and _var(node.node, base)
    return False


def _attr(node, name, key):
    import jinja2.nodes as n
    if isinstance(node, n.Getitem):
        return _var(node.node, name) and isinstance(node.arg, n.Const) and node.arg.value == key
    if isinstance(node, n.Getattr):
        return _var(node.node, name) and node.attr == key
    return False


def _elems(node, name, key=None):
    import jinja2.nodes as n
    if isinstance(node, n.Filter):
        return node.node is not None and _elems(node.node, name, key)
    if isinstance(node, n.Test):
        return _elems(node.node, name, key)
    if isinstance(node, n.Getitem) and isinstance(node.arg, n.Slice):
        return _elems(node.node, name, key)
    return _attr(node, name, key) if key else _var(node, name)


def _dotted(node):
    import jinja2.nodes as n
    if isinstance(node, n.NSRef):
        return f"{node.name}.{node.attr}"
    parts = []
    while isinstance(node, n.Getattr):
        parts.append(node.attr)
        node = node.node
    return ".".join([node.name, *reversed(parts)]) if isinstance(node, n.Name) else None


def _aliases_of(root, name):
    """vLLM's _iter_nodes_assign_var_or_elems: the variable and every variable assigned from it."""
    import jinja2.nodes as n
    out, seen, todo = [name], {name}, [name]
    while todo:
        cur = todo.pop(0)
        for a in root.find_all(n.Assign):
            lhs, rhs, found = a.target, a.node, []
            if isinstance(lhs, n.Name) and _elems(rhs, cur):
                found = [lhs.name]
            elif (isinstance(lhs, n.Name) and isinstance(rhs, n.Call) and isinstance(rhs.node, n.Name)
                  and rhs.node.name == "namespace"):
                found = [f"{lhs.name}.{kw.key}" for kw in rhs.kwargs if _elems(kw.value, cur)]
            elif isinstance(lhs, (n.Getattr, n.NSRef)) and _elems(rhs, cur):
                found = [d for d in [_dotted(lhs)] if d]
            for alias in found:
                out.append(alias)
                if alias not in seen:
                    seen.add(alias)
                    todo.append(alias)
    return out


def vllm_takes_parts(template, env):
    """vLLM's _detect_content_format: True for "openai", False for "string" (and for a template it
    cannot parse, its default)."""
    import jinja2.nodes as n
    try:
        root = env.parse(template)
    except Exception:  # noqa: BLE001
        return False
    messages = _aliases_of(root, "messages")
    message_vars = [loop.target.name for loop in root.find_all(n.For)
                    if isinstance(loop.target, n.Name) and any(_elems(loop.iter, v) for v in messages)]
    params_by_loop, in_macros = {}, set()
    for macro in root.find_all(n.Macro):
        names, params = {a.name for a in macro.args}, set()
        for call in root.find_all(n.Call):
            if isinstance(call.node, n.Name) and call.node.name == macro.name:
                for i, arg in enumerate(call.args):
                    if i < len(macro.args) and any(_elems(arg, v, "content") for v in message_vars):
                        params.add(macro.args[i].name)
                for kw in call.kwargs:
                    if kw.key in names and any(_elems(kw.value, v, "content") for v in message_vars):
                        params.add(kw.key)
        for loop in macro.find_all(n.For):
            in_macros.add(id(loop))
            if params:
                params_by_loop[id(loop)] = params
    for loop in root.find_all(n.For):
        if any(_elems(loop.iter, v, "content") for v in message_vars):
            return True
        if isinstance(loop.iter, n.Name) and loop.iter.name in params_by_loop.get(id(loop), ()):
            return True
        if id(loop) not in in_macros and isinstance(loop.iter, n.Name) and loop.iter.name == "content":
            return True
    return False


def sglang_takes_parts(template, env):
    """SGLang's detect_jinja_template_content_format: True for "openai"."""
    import jinja2.nodes as n
    if any(word in template for word in ("image", "audio", "video", "vision")):
        return True
    try:
        root = env.parse(template)
    except Exception:  # noqa: BLE001
        return False
    return any(_elems(loop.iter, v, "content") for loop in root.find_all(n.For) for v in ("message", "msg", "m"))


def content_forms(template, engine, env):
    """The forms a message's content reaches this template in: False a string, True a list of parts."""
    if engine == "vllm":
        return [vllm_takes_parts(template, env)]
    return [False, True] if sglang_takes_parts(template, env) else [False]


TOOLS = [{"type": "function", "function": {
    "name": "read", "description": "Read a file of the project.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def messages(roles, parts=False):
    """A conversation of these roles, with a tool call where the shape has one; each content a string,
    or with parts, a list of text parts, as vLLM hands a template that takes OpenAI's content format."""
    text = (lambda t: [{"type": "text", "text": t}]) if parts else (lambda t: t)
    out = []
    for i, role in enumerate(roles):
        if role == "assistant_tool_call":
            out.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "read", "arguments": {"path": "hello.txt"}}}]})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": f"call_{i - 1}", "content": text("7351")})
        else:
            out.append({"role": role, "content": text(f"A {role} message, number {i}.")})
    return out


def render(env, template, roles, tools, parts=False):
    """None when the template renders this conversation, else why it does not."""
    try:
        env.from_string(pick(template, tools)).render(messages=messages(roles, parts), tools=tools,
                                                      add_generation_prompt=True, bos_token="", eos_token="")
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
    failed = renders = 0
    for name, repo, rev, patch in lane_specs():
        template = served(repo, rev, fetch(repo, rev), patch)
        engine = "vllm" if name in VLLM_LANES else "sglang"
        how = "as published" if patch is None else f"patched ({patch.get('patches', 'all four fixes')})"
        forms = sorted({p for tools in (None, TOOLS) for p in content_forms(pick(template, tools), engine, env)})
        print(f"{name}: {repo}@{rev[:12]}, {how}, {engine}, content as "
              + " and ".join("parts" if p else "a string" for p in forms))
        for shape in shapes:
            for tools in (None, TOOLS):
                for parts in content_forms(pick(template, tools), engine, env):
                    renders += 1
                    why = render(env, template, shape["roles"], tools, parts)
                    if why:
                        failed += 1
                        print(f"  REFUSED {shape['client']}{' with tools' if tools else ''}"
                              f"{', content as parts' if parts else ''}: {why}")
    print(f"{renders} renders, {failed} refusal(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
