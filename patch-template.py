#!/usr/bin/env python3
"""Patch RadixArk/Qwen3.8-27B-NVFP4's chat template for agentic clients (Claude Code).

Five surgical fixes, no behavior change otherwise:
  1. reasoning_effort 'max'/'high' -> 'xhigh' (clients like Claude Code send "max";
     the stock template only accepts xhigh/medium/low and raises a 500 otherwise).
  2. Mid-conversation system messages -> rendered as <system-reminder> blocks
     (Claude Code injects system-reminders after turn 1; the stock template raises
     'System message must be at the beginning').
  3. The "lean" effort level, the default this box measured (LEAN.md).
  4. The refusal message names the levels and default the template really has.
  5. tool_reference content blocks (Claude Code 2.1.304's ToolSearch, Chrome's
     MCP tools among them) -> a text marker before the tool's result. The
     stock template ends its content loop with raise_exception('Unexpected item type
     in content.'), so every request carrying the tool search's result was a 500.

The 'minimal' -> 'low' mapping (an OpenAI effort tier) was contributed
by forum user helge: https://forums.developer.nvidia.com/t/380257/10

Usage: patch-template.py [--patches=NAMES] <hf_cache_dir> <output_path> [revision] [repo]
--patches=system-reminder (comma-separated names) applies those fixes alone, for a template with
no reasoning effort of its own; without it, all five. An option of the call, not a variable of the
environment: TEMPLATE_PATCHES, which v1.40.2 read as first pushed, reached every caller, and one
left in a shell made ./install.sh write the 27B's template without its effort fixes (found in
review, 2026-10-24).
Idempotent: succeeds if the patches are already applied. When a revision is
given (sha or ref name like 'main'), the template is taken from that exact
snapshot; otherwise the most recently modified snapshot is used. The optional
repo arg (default: RadixArk/Qwen3.8-27B-NVFP4) selects which cached repo to
read the template from.
"""
import glob
import os
import sys

EFFORT_ANCHOR = (
    "    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
    "    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
)
EFFORT_PATCHED = (
    "    {%- set resolved_reasoning_effort = reasoning_effort|default('@@DEF@@') %}\n"
    "    {%- if resolved_reasoning_effort in ('max', 'high') %}\n"
    "        {%- set resolved_reasoning_effort = 'xhigh' %}\n"
    "    {%- elif resolved_reasoning_effort == 'minimal' %}\n"
    "        {%- set resolved_reasoning_effort = 'low' %}\n"
    "    {%- endif %}\n"
    "    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low', 'lean') %}"
)

# -- Patch 3: the "lean" effort level, and it becomes the default -------------
# Qwen ships three levels and defaults to the most expensive one. Measured on
# this box (Qwen3.8-27B NVFP4 on SGLang, 7,008 runs; method and tables in
# LEAN.md):
#
#   On 364 public problems (HumanEval 164 + GSM8K 200), xhigh costs 3.19x the
#   thinking tokens of medium [2.73, 3.69] and scores 2.2 points LOWER
#   (McNemar p=0.057). On HumanEval alone it is 4.3 points lower at p=0.039,
#   and five of its ten failures there are truncations: it spent the whole
#   budget thinking and returned nothing.
#
#   On 58 underspecified requests, xhigh spends 2,478 thinking tokens and 237
#   seconds each against 420 and 73 for medium, truncates 10.3% of the time and
#   returns an empty answer 6.9% of the time.
#
# "lean" is 74 words that cost 0.436x the thinking tokens of medium
# [0.40, 0.47] on those 58 requests with no measured quality change, and 0.047x
# of xhigh [0.04, 0.06] with 17.2 points MORE usable answers [+10.9, +24.1].
#
# Qwen's three levels are left byte-identical: ask for medium and you get
# Qwen's medium, so numbers stay comparable with everyone else's. Only the
# DEFAULT moves, which is what every client that sends nothing receives.
# LEAN_DEFAULT=0 keeps Qwen's xhigh default while still offering the level.
LEAN_INSTRUCTIONS = "Answer immediately, with no reasoning, whenever the request asks for something you can simply write down: a rename, a reformat, a definition, a lookup, a single concrete edit. Reason only when producing the answer needs steps you cannot skip.\n\nIf the request is underspecified, pick the most common sensible interpretation, state it in one line, and proceed. If you notice yourself checking something twice, or weighing the same options again, stop there and commit."
# What a template that offers the lean level holds, and one written before v1.13 does not.
LEAN_MARKER = "'lean' %}"
LEAN_ANCHOR = "    {%- if resolved_reasoning_effort == 'xhigh' %}\n"
LEAN_PATCHED = (
    "    {%- if resolved_reasoning_effort == 'lean' %}\n"
    "        {%- set reasoning_instructions = '" + LEAN_INSTRUCTIONS + "' %}\n"
    "    {%- elif resolved_reasoning_effort == 'xhigh' %}\n"
)

# Patch 4: the refusal message must name what is actually supported, or a client
# that sends a bad value is told to use a default that is no longer the default.
MSG_ANCHOR = (
    "        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ "
    "'. Supported types are xhigh (default), medium, and low.') }}"
)
MSG_PATCHED = (
    "        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ "
    "'. Supported types are @@LEVELS@@.') }}"
)
# the refusal names the default the template really has (it said "lean (default)" on a box
# installed with LEAN_DEFAULT=0, whose default is xhigh)
MSG_LEVELS = {True: "lean (default), xhigh, medium, and low", False: "lean, xhigh (default), medium, and low"}

SYSTEM_ANCHOR = (
    "    {%- if message.role == \"system\" %}\n"
    "        {%- if not loop.first %}\n"
    "            {{- raise_exception('System message must be at the beginning.') }}\n"
    "        {%- endif %}"
)
SYSTEM_PATCHED = (
    "    {%- if message.role == \"system\" %}\n"
    "        {%- if not loop.first %}\n"
    "            {{- '<|im_start|>user\\n<system-reminder>\\n' + content + '\\n</system-reminder><|im_end|>' + '\\n' }}\n"
    "        {%- endif %}"
)

# Patch 5: Claude Code 2.1.304's ToolSearch (its tool search for MCP servers) answers a
# tool call with a tool_result made of tool_reference blocks, not text. SGLang's Anthropic
# endpoint hands those to the template as {"type": "tool_reference", "name": ...}; the stock
# content loop knows only image, video and text and raises. The reference names the tools the
# search loaded, whose schemas the next request carries, so its name is all the model needs
# here (measured 2026-10-25: a real session with 10 Chrome DevTools references got 500 on
# every turn, and the same block reproduced it alone; a text tool_result passed).
TOOL_REF_ANCHOR = (
    "            {%- elif 'text' in item %}\n"
    "                {{- item.text }}\n"
    "            {%- else %}\n"
    "                {{- raise_exception('Unexpected item type in content.') }}\n"
)
TOOL_REF_PATCHED = (
    "            {%- elif 'text' in item %}\n"
    "                {{- item.text }}\n"
    "            {%- elif item.type == 'tool_reference' %}\n"
    "                {{- '[tool available: ' ~ (item.name or item.tool_name or 'unknown') ~ ']' }}\n"
    "            {%- else %}\n"
    "                {{- raise_exception('Unexpected item type in content.') }}\n"
)
TOOL_REF_MARKER = "item.type == 'tool_reference'"
# A template is left as it is only when it raises nothing at all: a guard on the name of the macro
# skipped the fix for a template that raises in a renamed macro or inline, and one on the error's words
# for a template that words it otherwise, and the install went on with every turn after a ToolSearch a
# 500 again (code reviews, 2026-10-26). One that raises anywhere the fix cannot reach stops the install.
TOOL_REF_RAISE = "raise_exception("


# The five fixes by name, in the order they are applied (--patches picks among them).
PATCHES = ("reasoning_effort", "lean", "effort-message", "system-reminder", "tool-reference")
EFFORT_PATCHES = {"reasoning_effort", "lean", "effort-message"}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--patches=")]
    asked_patches = [a[len("--patches="):] for a in sys.argv[1:] if a.startswith("--patches=")]
    if len(args) not in (2, 3, 4) or len(asked_patches) > 1:
        sys.exit(__doc__)
    hf_cache, out_path = args[0], args[1]
    revision = args[2] if len(args) >= 3 else None
    repo = args[3] if len(args) == 4 else "RadixArk/Qwen3.8-27B-NVFP4"
    repo_dir = f"{hf_cache}/hub/models--{repo.replace('/', '--')}"
    chosen = None
    if revision:
        ref_file = f"{repo_dir}/refs/{revision}"
        if os.path.isfile(ref_file):  # ref name (e.g. 'main') -> resolve to the sha
            revision = open(ref_file, encoding="utf-8").read().strip()
        cand = f"{repo_dir}/snapshots/{revision}/chat_template.jinja"
        if os.path.isfile(cand):
            chosen = cand
        else:
            print(f"note: pinned revision {revision[:12]} has no chat_template.jinja snapshot, "
                  "falling back to the newest one")
    if chosen is None:
        hits = glob.glob(f"{repo_dir}/snapshots/*/chat_template.jinja")
        if not hits:
            sys.exit(f"chat_template.jinja not found under {hf_cache}. Run the checkpoint download first")
        chosen = max(hits, key=os.path.getmtime)
        if len(hits) > 1:
            print(f"note: {len(hits)} snapshots present, using the most recent: {chosen.split('/')[-2][:12]}")
    tpl = open(chosen, encoding="utf-8").read()

    # The fixes this lane asks for: all five unless it says otherwise, as the Qwen3.8 27B and
    # flash templates carry the anchors of all five. A checkpoint whose template has no reasoning
    # effort of its own asks for system-reminder alone (Ornith 1.5 122B, 2026-10-24: the proxy
    # gives it its effort as enable_thinking). A fix it asks for whose anchor is gone still stops
    # the install, as before.
    wanted = [p.strip() for p in (asked_patches[0] if asked_patches else ",".join(PATCHES)).split(",") if p.strip()]
    unknown = sorted(set(wanted) - set(PATCHES))
    if unknown or not wanted:
        sys.exit(f"--patches names {', '.join(unknown) or 'no fix'}: the fixes are {', '.join(PATCHES)}")

    # LEAN_DEFAULT=0 installs the level without making it the default, for a box
    # that wants Qwen's shipped behaviour until it has run its own numbers. Unset, the
    # template already installed decides: the choice was read from the environment of
    # every run, so a box installed with LEAN_DEFAULT=0 went back to lean at its next
    # ./install.sh or cockpit Switch, neither of which passes it (found in review,
    # 2026-09-24). LEAN.md presents it as a kept choice.
    # A template can only keep a choice it was able to make: one written before v1.13 has
    # no lean level and defaults to xhigh, the one default there was, so it chose nothing,
    # and read as LEAN_DEFAULT=0 it kept a box updated from there on xhigh without anyone
    # asking for it (a user's box, 2026-10-05: every bench.sh probe measured other text).
    # Only a template that offers lean and still defaults to xhigh was installed with
    # LEAN_DEFAULT=0.
    effort_patched = msg_patched = ""
    if EFFORT_PATCHES & set(wanted):
        lean_env = os.environ.get("LEAN_DEFAULT")
        installed, why = "", ""
        if lean_env is not None:
            lean_default = lean_env != "0"
        else:
            try:
                with open(out_path, encoding="utf-8") as f:
                    installed = f.read()
            except OSError:
                installed = ""
            has_lean = LEAN_MARKER in installed
            lean_default = not (has_lean and "reasoning_effort|default('xhigh')" in installed)
            if installed:
                why = (" (kept from the installed template)" if has_lean
                       else " (the installed template predates the lean level and chose nothing)")
        effort_patched = EFFORT_PATCHED.replace("@@DEF@@", "lean" if lean_default else "xhigh")
        msg_patched = MSG_PATCHED.replace("@@LEVELS@@", MSG_LEVELS[lean_default])
        print(f"default reasoning effort: {'lean' if lean_default else 'xhigh'}{why}")

    for name, anchor, patched, marker in (
        ("reasoning_effort", EFFORT_ANCHOR, effort_patched, "'minimal'"),
        ("lean", LEAN_ANCHOR, LEAN_PATCHED, LEAN_MARKER),
        ("effort-message", MSG_ANCHOR, msg_patched, "Supported types are lean"),
        ("system-reminder", SYSTEM_ANCHOR, SYSTEM_PATCHED, "<system-reminder>"),
        ("tool-reference", TOOL_REF_ANCHOR, TOOL_REF_PATCHED, TOOL_REF_MARKER),
    ):
        if name not in wanted:
            continue
        if anchor in tpl:
            tpl = tpl.replace(anchor, patched, 1)
            print(f"patch '{name}': applied")
        elif marker in tpl:
            print(f"patch '{name}': already present upstream, nothing to do")
        elif name == "tool-reference" and TOOL_REF_RAISE not in tpl:
            print("patch 'tool-reference': not needed (this template raises nothing)")
        else:
            sys.exit(
                f"patch '{name}': anchor not found and fix not present: the upstream "
                "template changed. Please open an issue with the template revision."
            )

    # The engine reads this file when it starts, and install.sh keeps a running engine only
    # when nothing it reads was written after it started: an identical template is left as
    # it is, mtime included, or every run would make the next one restart the engine.
    try:
        with open(out_path, encoding="utf-8") as f:
            if f.read() == tpl:
                print(f"patched template unchanged: {out_path}")
                return
    except OSError:
        pass
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tpl)
    print(f"patched template written to {out_path}")


if __name__ == "__main__":
    main()
