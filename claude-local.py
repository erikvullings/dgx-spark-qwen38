#!/usr/bin/env python3
"""claude-local: Claude Code on the lane this box serves (discussion #59, 2026-10-23).

Usage: claude-local [claude's own arguments...]     (installed in ~/.local/bin by install.sh)

It sets, for the one process it starts and nothing else (your ~/.claude is left alone):
the proxy as Anthropic's endpoint, the box's API key, read from its file at each launch and
handed over in the environment only (never on a command line, where ps shows it), the model
the box serves, and the two numbers Claude Code's context is ruled by, then runs `claude`.

The numbers follow the lane that serves. The installer and the cockpit fit opencode's
limits (`limit.context` and `limit.output` in ~/.config/qwen38/opencode.json) to the lane
that serves, its pool, its window and the proxy's one-prompt ceiling, at every install and
every Load. Claude Code compacts on a rule of its own, measured against Claude Code 2.1.304
on 2026-10-23 (as on 2.1.239 on 2026-08-21) for a model whose name is not a Claude one:

    effective window = CLAUDE_CODE_MAX_CONTEXT_TOKENS - min(CLAUDE_CODE_MAX_OUTPUT_TOKENS, 20,000)
    it compacts at the effective window - 13,000, through its reactive path for an unknown
    model (it sends the turn, and compacts if the server says the prompt is too long, which
    the proxy says), and it compacts before sending at the effective window - 3,000

So the longest prompt it ever sends is CLAUDE_CODE_MAX_CONTEXT_TOKENS - 23,000 (its output
setting is 32,000 unless you set it), and it asks for max_tokens on top. That prompt is set to
opencode's own context limit on this lane, with opencode's answer budget. opencode itself may
reach one worst step further (oc-fit-limits.py), and that step is what is left over here, for
what Claude Code adds to a prompt and only estimates: whatever fits opencode fits it too.

CLAUDE_CODE_ATTRIBUTION_HEADER is not set: with and without it, Claude Code on the Ornith
lane reused 36 to 38% of its prompts from the cache (2026-10-22), so it changes nothing here.
"""
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", Path.home() / ".config/qwen38"))
DEFAULT_BASE = "http://127.0.0.1:30001"
# What fits every lane's window (262,144 tokens on the smallest) and every pool this repo has
# measured, for a box whose opencode limits say nothing about the lane that serves.
FALLBACK = (205_000, 32_000)
# Claude Code's own output setting, as it reads it (its default when unset)
CC_OUTPUT_DEFAULT = 32_000


def say(msg: str) -> None:
    print(f"claude-local: {msg}", file=sys.stderr)


def read_key() -> str:
    try:
        return (CONFIG_DIR / "api-key").read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        say(f"no API key in {CONFIG_DIR}/api-key: run ./install.sh first")
        raise SystemExit(1)


def opencode_config() -> dict:
    try:
        return json.loads((CONFIG_DIR / "opencode.json").read_text())
    except (OSError, ValueError):
        return {}


def proxy_base(cfg: dict) -> str:
    """The proxy, as the box's opencode config reaches it (the installer writes it with the
    proxy's port), unless CLAUDE_LOCAL_URL names another."""
    if os.environ.get("CLAUDE_LOCAL_URL"):
        return os.environ["CLAUDE_LOCAL_URL"].rstrip("/")
    for prov in (cfg.get("provider") or {}).values():
        url = ((prov or {}).get("options") or {}).get("baseURL") or ""
        if url:
            return url.rstrip("/").removesuffix("/v1")
    return DEFAULT_BASE


def served_model(base: str, key: str) -> tuple[str, int]:
    """The model the proxy serves, and its window when the engine says it (vLLM does)."""
    req = urllib.request.Request(base + "/v1/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode()).get("data") or []
    except (urllib.error.URLError, OSError, ValueError) as e:
        say(f"no lane answers at {base} ({type(e).__name__}): load one from the cockpit first")
        raise SystemExit(1)
    if not data or not data[0].get("id"):
        say(f"{base}/v1/models lists no model: load a lane from the cockpit first")
        raise SystemExit(1)
    return data[0]["id"], int(data[0].get("max_model_len") or 0)


def opencode_limits(cfg: dict, model: str) -> tuple[int, int] | None:
    for prov in (cfg.get("provider") or {}).values():
        limit = (((prov or {}).get("models") or {}).get(model) or {}).get("limit") or {}
        context, output = limit.get("context"), limit.get("output")
        if isinstance(context, int) and isinstance(output, int) and context > 0 and output > 0:
            return context, output
    return None


def claude_pair(limits: tuple[int, int] | None, window: int, cc_output: int) -> tuple[int, int, str]:
    """(CLAUDE_CODE_MAX_CONTEXT_TOKENS, max_tokens, where they come from)."""
    block = min(cc_output, 20_000) + 3_000      # its longest prompt is the context minus this
    if limits:
        context, output = limits
        return context + block, output, "this lane's opencode limits"
    context, output = FALLBACK
    if window and context - block + output > window:     # a lane smaller than every one today
        context = max(0, window - output + block)
    return context, output, "the pair that fits every lane (no opencode limits for this lane)"


def main(argv: list[str]) -> int:
    claude = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    if not os.access(claude, os.X_OK):
        say("Claude Code is not installed (no `claude` on PATH). Install it as Anthropic's docs say "
            "(https://code.claude.com/docs/en/setup), then run claude-local again.")
        return 127
    key = read_key()
    cfg = opencode_config()
    base = proxy_base(cfg)
    model, window = served_model(base, key)
    cc_output = int(os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS") or CC_OUTPUT_DEFAULT)
    context, output, why = claude_pair(opencode_limits(cfg, model), window, cc_output)
    env = dict(os.environ)
    env.update({
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_API_KEY": "",
        # any name reaches the lane that serves; this one is the lane's own
        "ANTHROPIC_MODEL": env.get("ANTHROPIC_MODEL") or model,
        # a long prefill or a long answer is not a dead stream
        "API_TIMEOUT_MS": env.get("API_TIMEOUT_MS") or "3600000",
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS": env.get("CLAUDE_STREAM_IDLE_TIMEOUT_MS") or "1800000",
        "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": env.get("CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS") or "1800000",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") or "1",
    })
    # yours win: a context or an answer budget you set is kept as it is
    env.setdefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(context))
    env.setdefault("CLAUDE_CODE_EXTRA_BODY", json.dumps({"max_tokens": output}))
    if sys.stderr.isatty():
        mine = "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in os.environ or "CLAUDE_CODE_EXTRA_BODY" in os.environ
        say(f"{env['ANTHROPIC_MODEL']} through {base}, context {env['CLAUDE_CODE_MAX_CONTEXT_TOKENS']}, "
            f"extra body {env['CLAUDE_CODE_EXTRA_BODY']} ({'yours, kept' if mine else why})")
    os.execve(claude, [claude, *argv], env)
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
