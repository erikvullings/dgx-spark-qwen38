#!/usr/bin/env python3
"""The Nemotron lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the
two cannot drift.

Usage: oc-nemotron.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"nemotron": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# Bolt's chat template (NVIDIA's model card) reads enable_thinking and medium_effort, not a
# reasoning effort: thinking is on by default, at its full depth. The levels opencode offers
# map onto those two, as the box decided (2026-10-15): none turns the reasoning off, low and
# medium ask the template's shorter "efficient" reasoning, high is its default.
VARIANTS = {
    "none": {"chat_template_kwargs": {"enable_thinking": False}},
    "low": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}},
    "medium": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}},
    "high": {"chat_template_kwargs": {"enable_thinking": True}},
}


def provider(base_url, key_file, ctx, out, name):
    """The provider block. Bolt reads images (and videos), so opencode may attach one."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Nemotron 3.5 Bolt (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"nemotron-3.5-bolt": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: dict(v) for lvl, v in VARIANTS.items()},
            "attachment": True,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        }},
    }


def main():
    if len(sys.argv) != 6:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    base_url, key_file, ctx, out, name = sys.argv[1:6]
    try:
        ctx, out = int(ctx), int(out)
    except ValueError:
        print("ctx and out are token counts", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"provider": {"nemotron": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
