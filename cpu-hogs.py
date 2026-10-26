#!/usr/bin/env python3
"""The processes that hold a CPU core outside the serving engine.

On a GB10 the CPU and the GPU share one power budget, so a process that spins a core costs the
engine speed and heat, and shows only as a load average: on another DGX Spark, in October 2026,
three command-line calls that old sessions left behind had each held a core for four to six days.
bench.sh names such processes before it measures, from one short window taken while the engine
waits for its first request.

The engine's own processes are left out: those of the lanes' containers (docker names qwen38-*).
A process another lane runs outside docker (the image and video lanes) is named, as it shares the
box with the lane measured.

Usage: cpu-hogs.py [--window SECONDS] [--floor SHARE]
  one line per process at or above FLOOR of a core over the window, the busiest first:
  pid, share of a core, name and command line, separated by tabs; nothing when none.
"""
import argparse
import os
import subprocess
import sys
import time


def ticks(proc="/proc"):
    """{pid: (CPU ticks used so far, user and system, every thread), name} of the processes readable now."""
    out = {}
    for d in os.listdir(proc):
        if not d.isdigit():
            continue
        try:
            with open(f"{proc}/{d}/stat") as f:
                raw = f.read()
        except OSError:
            continue
        # the name is in parentheses and may hold spaces and parentheses itself: the fields
        # after it start past the last one
        end = raw.rindex(")")
        fields = raw[end + 2:].split()
        out[int(d)] = (int(fields[11]) + int(fields[12]), raw[raw.index("(") + 1:end])
    return out


def engine_ids():
    """The full IDs of the lanes' running containers, or [] when docker does not answer."""
    try:
        r = subprocess.run(["docker", "ps", "--no-trunc", "--filter", "name=^qwen38-", "--format", "{{.ID}}"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return r.stdout.split() if r.returncode == 0 else []


def cgroup(pid, proc="/proc"):
    try:
        with open(f"{proc}/{pid}/cgroup") as f:
            return f.read()
    except OSError:
        return ""


def command(pid, proc="/proc"):
    try:
        with open(f"{proc}/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def busy(before, after, seconds, hz, floor, skip=lambda pid: False):
    """(pid, share of a core, name) of the processes at or above FLOOR between two readings of
    ticks() taken SECONDS apart, the busiest first. A process present in one reading only is left
    out: it started or ended inside the window."""
    found = []
    for pid, (used, name) in after.items():
        if pid not in before or skip(pid):
            continue
        share = (used - before[pid][0]) / hz / seconds
        if share >= floor:
            found.append((pid, share, name))
    return sorted(found, key=lambda x: -x[1])


def hogs(window=3.0, floor=0.5, proc="/proc"):
    """The processes outside the engine at or above FLOOR of a core over WINDOW seconds."""
    hz = os.sysconf("SC_CLK_TCK")
    before, t0 = ticks(proc), time.monotonic()
    time.sleep(window)
    after, seconds = ticks(proc), time.monotonic() - t0
    ids = engine_ids()
    me = {os.getpid(), os.getppid()}
    return [(pid, share, name, command(pid, proc))
            for pid, share, name in busy(before, after, seconds, hz, floor,
                                         lambda pid: pid in me or any(i in cgroup(pid, proc) for i in ids))]


def printable(text):
    """A field of the tab-separated output: no tab or newline to split the line (bench.sh keeps lines of four
    fields, and a busy process with a tab in its argv vanished from its report), and no control character a
    terminal would act on (an argv can carry an escape sequence) (verification, 2026-10-26)."""
    return "".join(" " if ord(c) < 32 or 127 <= ord(c) < 160 else c for c in text)


def main(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--floor", type=float, default=0.5)
    a = ap.parse_args(argv)
    for pid, share, name, cmd in hogs(a.window, a.floor):
        print(f"{pid}\t{share:.2f}\t{printable(name)}\t{printable(cmd)[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
