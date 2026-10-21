#!/usr/bin/env python3
"""The Ornith 1.5 122B lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the two
cannot drift.

Usage: oc-ornith.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"ornith-122b": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# Ornith's chat template (its checkpoint's, served as published) reads enable_thinking and no
# reasoning effort: thinking is on by default, at its full depth. The box maps the levels onto it,
# as decided on 2026-10-21: none turns the reasoning off, every other level keeps it whole. So two
# variants, the default first, as a picker lists it.
VARIANTS = {
    "high": {"chat_template_kwargs": {"enable_thinking": True}},
    "none": {"chat_template_kwargs": {"enable_thinking": False}},
}


def provider(base_url, key_file, ctx, out, name):
    """The provider block. The model reads images, so opencode may attach one."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Ornith 1.5 122B (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"ornith-1.5-122b": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: json.loads(json.dumps(v)) for lvl, v in VARIANTS.items()},
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
    print(json.dumps({"provider": {"ornith-122b": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
