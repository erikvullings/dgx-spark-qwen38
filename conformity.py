#!/usr/bin/env python3
"""What this release recommends for a box, and which values only an older release made the default.

An update kept, as if the operator had chosen them, values that were nothing but the default
of the release that first installed the box: the engine on every interface (the default until
v1.17.0), the native 262144 window (until v1.12.1), the flash lane's context tier (until
v1.19.0) and the BF16 draft at depth 8 (until v1.9.0). A unit cannot say which is which, and
"Keeping the installed engine bind: 0.0.0.0" read as a choice on a box where nobody had ever
made one (found while cataloguing every way a box departs from a fresh one, 2026-10-16).

This module holds the facts that tell them apart, for install.sh and the cockpit (since
v1.36.0, 2026-10-17): what a box installed today gets, read from install.sh itself so it
cannot drift from it, and the four old defaults with the instant their release moved them.

Standard library only: install.sh runs it with the system python3, and the cockpit loads it
the way it loads engine-inputs.py.
"""
import re
from collections import namedtuple
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Every setting this module compares, by the name an operator passes to set it, so that the way
# to keep a value is always NAME=value ./install.sh. A lane's target is not a variable of its
# own (MODEL_CHOICE picks the lane too), hence target.27b and target.flash; DRAFT2 is the 27B
# drafter as a whole (repository, quantization, depth), because the old default is the three
# together: the same repository at depth 16 was somebody's choice.
KNOBS = ("target.27b", "target.flash", "DRAFT2", "CONTEXT_MODE", "ENGINE_BIND", "PROXY_BIND",
         "PORT", "PROXY_PORT", "HF_CACHE", "PLE_DIR", "FLASH_TIER", "SPEC_TOKEN_MAP_SIZE",
         "FLASH_REPLAYSSM_SPEC")
# Settings that belong to the site rather than to a release: where the checkpoints live, which
# ports clients were given. Listed apart and never moved by an update (decided 2026-10-16).
SITE = ("PORT", "PROXY_PORT", "HF_CACHE", "PLE_DIR")

# A default an older release installed, and the instant the release that moved it was made.
# The instant is the committer date of the tagged commit, not its day: v1.16.0 and v1.17.0
# shipped 18 minutes apart, v1.12.0 and v1.12.1 54, and a box installed between them must not
# be classed on the wrong side. Checked 2026-10-16 with
#   git for-each-ref --format='%(*committerdate:iso-strict)%(committerdate:iso-strict)' refs/tags/<tag>
# `says` is the CHANGELOG's own words for the change, in that release's section
# (tests/test_conformity_recommended.py holds each row to it).
OldDefault = namedtuple("OldDefault", "knob value tag instant says")
OLD_DEFAULTS = (
    # z-lab's BF16 DFlash2 at depth 8: the drafter of v1.2.3 to v1.8.6 (their unit template)
    OldDefault("DRAFT2", "z-lab/Qwen3.8-27B-DFlash2 (unquant, D=8)", "v1.9.0",
               "2026-09-11T13:06:45+02:00", "the 27B lane drafts from a calibrated NVFP4 head, 16 deep"),
    # CONTEXT_MODE:-native until v1.12.0; empty since, which a 27B service install reads as 1m
    OldDefault("CONTEXT_MODE", "native", "v1.12.1",
               "2026-09-14T12:13:42+02:00", "the 1M window by default"),
    # --host 0.0.0.0 in the templates until v1.15.x, ENGINE_BIND:-0.0.0.0 in v1.16.0
    OldDefault("ENGINE_BIND", "0.0.0.0", "v1.17.0",
               "2026-09-22T13:19:42+02:00", "the engine answers on localhost only, and that is the default"),
    # FLASH_TIER:-context until v1.18.7
    OldDefault("FLASH_TIER", "context", "v1.19.0",
               "2026-10-01T02:07:06+02:00", "`FLASH_TIER` defaults to `concurrency`"),
)


def old_default_of(knob):
    """The old default row of a setting, or None when no release ever installed another default."""
    return next((o for o in OLD_DEFAULTS if o.knob == knob), None)


def old_default(knob, value):
    """The old default row when `value` is what an older release installed for `knob`, else None."""
    o = old_default_of(knob)
    return o if o is not None and o.value == value else None


def drafter(repo, quant, tokens):
    """The 27B drafter as one value, the way install.sh reads it back from a unit."""
    return f"{repo} ({quant}, D={tokens})"


def _install_text(repo_dir):
    return (Path(repo_dir) / "install.sh").read_text(encoding="utf-8")


def installer_defaults(repo_dir=REPO):
    """{VAR: default} for each `VAR="${VAR:-default}"` line of install.sh, the first one of each
    variable: the line a default is written on."""
    out = {}
    for m in re.finditer(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-([^}]*)\}"', _install_text(repo_dir), re.M):
        out.setdefault(m.group(1), m.group(2))
    return out


def targets(repo_dir=REPO):
    """{target: repository} for every MODEL_CHOICE install.sh accepts, from its own case block."""
    text = _install_text(repo_dir)
    repos = dict(re.findall(r'^([A-Z][A-Z0-9_]*_REPO)="([^"$]+)"', text, re.M))
    return {name: repos[var] for name, var in
            re.findall(r'^ {2}([a-z0-9-]+)\)\s+MODEL_REPO="\$([A-Z][A-Z0-9_]*_REPO)"', text, re.M)}


def recommended(repo_dir=REPO, home=None):
    """{knob: value} a box installed today with no variable set ends up with. A default install.sh
    no longer writes the way this reads raises KeyError rather than recommend nothing."""
    home = str(home or Path.home())
    d = installer_defaults(repo_dir)

    def val(name):
        return d[name].replace("$HOME", home)

    port = val("PORT")
    proxy_port = val("PROXY_PORT")
    if proxy_port == "$((PORT+1))":
        proxy_port = str(int(port) + 1)
    flash_target = re.search(r"SECOND_LANE=flash;.*\bSECOND_TARGET=([a-z0-9-]+)$", _install_text(repo_dir), re.M)
    if not flash_target:
        raise KeyError("SECOND_TARGET of the flash lane")
    return {
        "target.27b": val("MODEL_CHOICE"),
        "target.flash": flash_target.group(1),
        "DRAFT2": drafter(val("DRAFT2_REPO"), val("DRAFT2_QUANT"), val("DRAFT2_TOKENS")),
        # empty in install.sh, and resolved to 1m on a 27B service install once the flags are
        # read (the test runs install.sh to check that this is what it resolves)
        "CONTEXT_MODE": "1m",
        "ENGINE_BIND": val("ENGINE_BIND"),
        "PROXY_BIND": val("PROXY_BIND"),
        "PORT": port,
        "PROXY_PORT": proxy_port,
        "HF_CACHE": val("HF_CACHE"),
        "PLE_DIR": val("PLE_DIR"),
        "FLASH_TIER": val("FLASH_TIER"),
        "SPEC_TOKEN_MAP_SIZE": val("SPEC_TOKEN_MAP_SIZE"),
        "FLASH_REPLAYSSM_SPEC": val("FLASH_REPLAYSSM_SPEC"),
    }
