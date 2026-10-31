#!/usr/bin/env python3
"""The Groot 65B A6B lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the two
cannot drift.

Usage: oc-groot.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"groot": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# Groot exposes enable_thinking as a boolean.
VARIANTS = {
    level: {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True,
                                     "reasoning_effort": level}}
    for level in ("low", "medium", "xhigh")
}
VARIANTS["none"] = {"chat_template_kwargs": {"enable_thinking": False}}



def provider(base_url, key_file, ctx, out, name):
    """The provider block. The model reads images, so opencode may attach one."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Groot 65B A6B (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"groot-65b-a6b": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: json.loads(json.dumps(v)) for lvl, v in VARIANTS.items()},
            "attachment": False,
            "modalities": {"input": ["text"], "output": ["text"]},
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
    print(json.dumps({"provider": {"groot": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
