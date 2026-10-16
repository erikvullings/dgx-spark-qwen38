#!/usr/bin/env python3
"""The Nemotron lanes' opencode providers, in one place.

install.sh's generator writes each into the opencode config once its lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the
two cannot drift. One provider per lane (nemotron for Bolt, nemotron-super for Super VL): each
lane serves its own model, with its own window.

Usage: oc-nemotron.py <base_url> <api-key file> <ctx> <out> <name> [nemotron|nemotron-super]
prints {"provider": {"<lane>": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# Bolt's and Super VL's chat templates (NVIDIA's model cards) read enable_thinking and
# medium_effort, not a reasoning effort: thinking is on by default, at its full depth. The levels
# opencode offers map onto those two, as the box decided (2026-10-15): none turns the reasoning
# off, low and medium ask the template's shorter "efficient" reasoning, high is its default.
VARIANTS = {
    "none": {"chat_template_kwargs": {"enable_thinking": False}},
    "low": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}},
    "medium": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}},
    "high": {"chat_template_kwargs": {"enable_thinking": True}},
}


# Each lane's provider title and served model.
LANES = {"nemotron": ("Nemotron 3.5 Bolt (DGX Spark)", "nemotron-3.5-bolt"),
         "nemotron-super": ("Nemotron 3.5 Super VL (DGX Spark)", "nemotron-3.5-super-vl")}


def provider(base_url, key_file, ctx, out, name, lane="nemotron"):
    """The provider block. Both models read images (and videos), so opencode may attach one."""
    title, model = LANES[lane]
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": title,
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {model: {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: dict(v) for lvl, v in VARIANTS.items()},
            "attachment": True,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        }},
    }


def main():
    if len(sys.argv) not in (6, 7) or (len(sys.argv) == 7 and sys.argv[6] not in LANES):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    base_url, key_file, ctx, out, name = sys.argv[1:6]
    lane = sys.argv[6] if len(sys.argv) == 7 else "nemotron"
    try:
        ctx, out = int(ctx), int(out)
    except ValueError:
        print("ctx and out are token counts", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"provider": {lane: provider(base_url, key_file, ctx, out, name, lane)}}, indent=2))


if __name__ == "__main__":
    main()
