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

A session starts in Claude Code's normal permission mode, which asks before each command, unless
you pass a mode of your own, and nothing of it goes to Anthropic (NO_ANTHROPIC_ENV below): the
model's requests go to the box, and Claude Code's own traffic to Anthropic is cut.

CLAUDE_CODE_ATTRIBUTION_HEADER is not set: with and without it, Claude Code on the Ornith
lane reused 36 to 38% of its prompts from the cache (2026-10-22), so it changes nothing here.
"""
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# The box's config folder, by a name of this repo's own: a CONFIG_DIR another tool exports was
# taken for it (found in review, 2026-10-24). The launcher install.sh writes sets it.
CONFIG_DIR = Path(os.environ.get("QWEN38_CONFIG_DIR") or Path.home() / ".config/qwen38")
DEFAULT_BASE = "http://127.0.0.1:30001"
# What fits every lane's window (262,144 tokens on the smallest), every pool measured on a 128 GB
# box and Bolt's on a 64 GB one, for a box whose opencode limits say nothing about the lane that
# serves. The 27B's pool on a 64 GB box is smaller: there the proxy refuses a prompt past it, and
# Claude Code compacts on that refusal.
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
    proxy's port), unless CLAUDE_LOCAL_URL names another; with no opencode config, the port the
    launcher was written with (QWEN38_PROXY_URL): the default one missed a box on another port
    (found in review, 2026-10-24)."""
    if os.environ.get("CLAUDE_LOCAL_URL"):
        return os.environ["CLAUDE_LOCAL_URL"].rstrip("/")
    for prov in (cfg.get("provider") or {}).values():
        url = ((prov or {}).get("options") or {}).get("baseURL") or ""
        if isinstance(url, str) and url:
            return url.rstrip("/").removesuffix("/v1")
    return (os.environ.get("QWEN38_PROXY_URL") or DEFAULT_BASE).rstrip("/")


# The probe goes to the box itself, never through an HTTP proxy the shell names: urllib follows
# http_proxy, and the box's key went out in clear to it unless no_proxy named 127.0.0.1 (found in
# review, 2026-10-24).
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def served_model(base: str, key: str) -> tuple[str, int]:
    """The model the proxy serves, and its window when the engine says it (vLLM does)."""
    req = urllib.request.Request(base + "/v1/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with DIRECT.open(req, timeout=10) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # an answer, so not a lane missing: a key this proxy refuses, or an engine booting
        why = "the box's key was refused (another box's proxy?)" if e.code in (401, 403) else "a lane may be starting"
        say(f"{base}/v1/models answered HTTP {e.code}: {why}")
        raise SystemExit(1)
    except (urllib.error.URLError, OSError, ValueError) as e:
        say(f"no lane answers at {base} ({type(e).__name__}): load one from the cockpit first")
        raise SystemExit(1)
    data = body.get("data") if isinstance(body, dict) else None
    first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    if not isinstance(first.get("id"), str) or not first["id"]:
        say(f"{base}/v1/models lists no model: load a lane from the cockpit first")
        raise SystemExit(1)
    window = first.get("max_model_len")
    return first["id"], window if isinstance(window, int) and window > 0 else 0


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


# Nothing of a claude-local session goes to Anthropic (Hasan, 2026-10-24): the model's requests,
# auto mode's classifier among them, go to the box, and Claude Code's other traffic is cut here.
# CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC cuts telemetry, error reports, /feedback, updates and
# feature flags (Remote Control with them, as it is off anyway for a base URL other than
# Anthropic's); these cut what it leaves on: the claude.ai connectors of a signed-in account, the
# official marketplace's registration, and auto mode's wait for checks the box's engine cannot
# make (its notice held the first command). These win over your shell's, as the endpoint and the
# key do, the first one included: Claude Code 2.1.304 reads it as set whatever its value but an empty
# one, so a shell's 0 kept the traffic off, but a reading of it as a switch would not (2026-10-24).
# Your own MCP servers stay: they are yours, and Anthropic hosts none of them.
NO_ANTHROPIC_ENV = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "CLAUDE_CODE_AUTO_MODE_SERVER": "0",
}
# WebFetch sends each hostname to api.anthropic.com before it fetches, whatever the provider: a
# setting, not a variable, so it goes on the session's command line, with the permission mode.
NO_ANTHROPIC_SETTINGS = {"skipWebFetchPreflight": True}
# A session starts in the normal mode, which asks before each command: an account set to auto mode
# had the box's own model judge which commands ran unasked (Hasan, 2026-10-24). One you pass wins,
# and --dangerously-skip-permissions stays yours to pass.
PERMISSION_FLAGS = ("--permission-mode", "--dangerously-skip-permissions")
# Claude Code's own commands take the environment and none of a session's flags.
# Claude Code's commands, their aliases (plugins, kill, upgrade) and the ones its help does not list
# (daemon, and what its binary reads as its first argument before any option: Remote Control's
# remote-control, rc, remote, sync and bridge, and self-hosted-runner): a session flag in front of one
# made it a prompt, or Remote Control's refusal, and `claude-local kill <name>` opened a session instead
# of stopping one, where a short id still passed (found in review, 2026-10-24;
# tests/fixtures/claude-code-commands.txt is Claude Code 2.1.304's own list).
CLAUDE_COMMANDS = frozenset({"agents", "attach", "auth", "auto-mode", "doctor", "gateway", "import", "install",
                             "logs", "mcp", "plugin", "plugins", "purge", "respawn", "rm", "setup-token", "stop",
                             "kill", "ultrareview", "update", "upgrade", "daemon", "remote-control", "rc",
                             "remote", "sync", "bridge", "self-hosted-runner"})


def session_flags(argv: list[str]) -> list[str]:
    """The flags a session gets ahead of yours: none for one of Claude Code's own commands."""
    if argv and argv[0] in CLAUDE_COMMANDS:
        return []
    given = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    flags = [] if given & set(PERMISSION_FLAGS) else ["--permission-mode", "default"]
    if "--settings" not in given:
        flags += ["--settings", json.dumps(NO_ANTHROPIC_SETTINGS)]
    return flags


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
    try:
        cc_output = int(os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS") or CC_OUTPUT_DEFAULT)
    except ValueError:
        say(f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={os.environ['CLAUDE_CODE_MAX_OUTPUT_TOKENS']} is not a number of tokens")
        return 2
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
        **NO_ANTHROPIC_ENV,
    })
    # the box itself, never through a proxy the shell names (Claude Code reads HTTPS_PROXY too)
    host = urllib.parse.urlsplit(base).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        for name in ("NO_PROXY", "no_proxy"):
            have = [h for h in env.get(name, "").split(",") if h.strip()]
            env[name] = ",".join(have + [h for h in ("127.0.0.1", "localhost") if h not in have])
    # yours win: a context or an answer budget you set is kept as it is; an extra body of yours
    # without max_tokens gets the lane's, as it otherwise dropped it (found in review, 2026-10-24)
    env.setdefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(context))
    try:
        extra = json.loads(env.get("CLAUDE_CODE_EXTRA_BODY") or "{}")
    except ValueError:
        extra = None
    if isinstance(extra, dict) and "max_tokens" not in extra:
        env["CLAUDE_CODE_EXTRA_BODY"] = json.dumps({**extra, "max_tokens": output})
    # Claude Code applies CLAUDE_CODE_MAX_CONTEXT_TOKENS to a model it does not know only
    if env["ANTHROPIC_MODEL"].lower().startswith("claude"):
        say(f"ANTHROPIC_MODEL={env['ANTHROPIC_MODEL']} names a Claude model: Claude Code then sizes the context "
            f"for that model, not for this lane; unset it to get the lane's ({model})")
    if sys.stderr.isatty():
        mine = "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in os.environ or "CLAUDE_CODE_EXTRA_BODY" in os.environ
        say(f"{env['ANTHROPIC_MODEL']} through {base}, context {env['CLAUDE_CODE_MAX_CONTEXT_TOKENS']}, "
            f"extra body {env['CLAUDE_CODE_EXTRA_BODY']} ({'yours, kept' if mine else why})")
    os.execve(claude, [claude, *session_flags(argv), *argv], env)
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
