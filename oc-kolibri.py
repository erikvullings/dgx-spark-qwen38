#!/usr/bin/env python3
"""The Kolibri lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that (a first install writes the config
first and installs the Kolibri lane at its end), so both take it from here and the two
cannot drift.

Usage: oc-kolibri.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"kolibri": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# Kolibri's own levels, read by its chat template (Aleph Alpha's model card): none turns its
# reasoning off, and no level at all is its default, high.
LEVELS = ("none", "low", "medium", "high")


def provider(base_url, key_file, ctx, out, name):
    """The provider block. Text only: Kolibri has no vision, and an image sent to it is
    refused, so opencode is told not to attach one."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Kolibri-1 (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"kolibri-1": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: {"chat_template_kwargs": {"reasoning_effort": lvl}} for lvl in LEVELS},
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
    print(json.dumps({"provider": {"kolibri": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
