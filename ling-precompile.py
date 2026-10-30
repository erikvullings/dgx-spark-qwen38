#!/usr/bin/env python3
"""Generate FlashInfer rules, release the owned model process, then compile without its weights.

Adapted from inclusionAI's DGX Spark Humming notebook (MIT, Ant Group and NVIDIA, 2026).
Only our own process group is signalled. No global process killing or cache deletion.
"""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import importlib.util
spec = importlib.util.spec_from_file_location("ling_launch", Path(__file__).with_name("ling-launch.py"))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)

ROOTS = [Path.home() / ".cache/flashinfer", Path.home() / ".cache/sglang/.cache/flashinfer"]


def find(name):
    hits = [p for root in ROOTS for p in root.glob("*/121a/cached_ops/fused_moe_120/" + name)]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def main():
    if find("fused_moe_120.so"):
        print("Ling FlashInfer kernel cache is already compiled.", flush=True)
        return
    ninja = shutil.which("ninja")
    if not ninja:
        sys.exit("ninja is missing from the Ling runtime")
    os.environ.update(launch.ENV)
    proc = subprocess.Popen(launch.command(), start_new_session=True)
    rules = None
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            rules = find("build.ninja")
            if rules:
                break
            if proc.poll() is not None:
                sys.exit("Ling exited before generating kernel build rules; inspect its logs.")
            time.sleep(1)
        if rules is None:
            sys.exit("Ling kernel rule generation timed out after 15 minutes.")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
    # Defunct descendants hold no model memory. PID 1 may still be reaping them.
    def live_group():
        for entry in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat = entry.read_text()
                fields = stat[stat.rindex(")") + 2:].split()
                if int(fields[2]) == proc.pid and fields[0] != "Z":
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False
    deadline = time.monotonic() + 10
    while live_group() and time.monotonic() < deadline:
        time.sleep(0.1)
    if live_group():
        sys.exit("Ling child processes still hold memory; refusing kernel compilation.")
    available = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    jobs = max(1, min(8, os.cpu_count() or 1, available // (10 * 1024**3)))
    subprocess.run([ninja, "-j" + str(jobs), "-C", str(rules.parent)], check=True, timeout=900)
    if find("fused_moe_120.so") is None:
        sys.exit("Compilation produced no expected SM121 kernel; refusing a model load.")


if __name__ == "__main__":
    main()
