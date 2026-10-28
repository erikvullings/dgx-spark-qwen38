#!/usr/bin/env python3
"""What a pair of DGX Spark (or GB10) boxes needs to know about each one, read-only (#45).

Run it on each box, no sudo needed, and paste what it prints:

    python3 ~/dgx-spark-qwen38/pair-probe.py                      # this box
    python3 ~/dgx-spark-qwen38/pair-probe.py --peer 192.168.100.2  # and the link to the other one

It reads the OS, the kernel, the GPU driver, this repo's installed release and lanes, the ConnectX-7
ports and their RDMA side, Docker, the time sync and the free disk, and, given --peer, how the other
box answers over which port. It changes nothing and starts nothing.

What it never prints: an API key or any file that holds one (it opens none), a MAC address, an IP
address (only its kind: private, link-local, shared, public), the host's name, your home directory
or user name. --json prints the same facts for a program.
"""
import argparse
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

HOME = str(Path.home())
REPO = Path(HOME) / "dgx-spark-qwen38"
CFG = Path(HOME) / ".config" / "qwen38"


def run(argv, timeout=8.0):
    """A command's output, or "" when it is missing, fails or takes too long. Never a shell."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def read(path, default=""):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return default


def ip_kind(addr):
    """private, link-local, shared (100.64/10: Tailscale, carrier NAT), loopback or public: never the
    address itself."""
    try:
        ip = ipaddress.ip_address(addr.split("/")[0])
    except ValueError:
        return "unknown"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
        return "shared"
    return "private" if ip.is_private else "public"


def masked(text, host=None):
    """Whatever a command said, without what could name the box or its owner: MAC addresses, valid IP
    addresses (their kind only), the home directory, and the host's name as a whole word (a name like
    "spark" must not eat "dgx-spark-qwen38")."""
    host = socket.gethostname() if host is None else host
    text = re.sub(r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b", "<mac>", text, flags=re.I)
    def ip(m):
        kind = ip_kind(m.group(0))
        return m.group(0) if kind == "unknown" else f"<{kind}>"      # a version like 580.178.04.1 stays
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?\b", ip, text)
    text = text.replace(HOME, "~")
    if host:
        text = re.sub(r"(?<![\w.-])" + re.escape(host) + r"(?![\w-])", "<host>", text)
    return text


def box(release="/etc/dgx-release"):
    # DGX OS appends a DGX_OTA_VERSION line at each update: the last one is the version it runs, not
    # DGX_SWBUILD_VERSION, the image it was installed from. DGX_SERIAL is in the same file, never shown.
    pairs = re.findall(r'^(\w+)="?([^"\n]*)"?$', read(release), re.M)
    ota = [v for k, v in pairs if k == "DGX_OTA_VERSION"]
    rel = dict(pairs)
    osr = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', read("/etc/os-release"), re.M))
    mem = re.search(r"MemTotal:\s+(\d+) kB", read("/proc/meminfo"))
    gpu = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).strip()
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", run(["nvidia-smi"]))
    return {
        "machine": f"{read('/sys/class/dmi/id/sys_vendor')} {read('/sys/class/dmi/id/product_name')}".strip(),
        "dgx_os": ota[-1] if ota else rel.get("DGX_SWBUILD_VERSION", ""),
        "platform": rel.get("DGX_PLATFORM", ""),
        "os": osr.get("PRETTY_NAME", ""),
        "kernel": os.uname().release,
        "arch": os.uname().machine,
        "memory_gib": round(int(mem.group(1)) / 2**20, 1) if mem else None,
        "gpu": gpu,
        "cuda": cuda.group(1) if cuda else "",
    }


def repo():
    try:
        inst = json.loads(read(CFG / "installed.json", "{}") or "{}")
    except ValueError:
        inst = {}
    git = lambda *a: run(["git", "-C", str(REPO), *a]).strip()
    units = run(["systemctl", "list-units", "--all", "--no-legend", "--plain", "qwen38-*"])
    lanes = {}
    for ln in units.splitlines():
        parts = ln.split()
        if len(parts) >= 4 and parts[0].endswith(".service"):
            enabled = run(["systemctl", "is-enabled", parts[0]]).strip()
            lanes[parts[0][:-8]] = f"{parts[2]}/{parts[3]}" + (" (boot)" if enabled == "enabled" else "")
    return {
        "release": inst.get("release", ""),
        "commit": (inst.get("commit") or "")[:7],
        "lane": inst.get("lane", ""),
        "checkout": git("rev-parse", "--short", "HEAD"),
        "branch": git("symbolic-ref", "-q", "--short", "HEAD"),
        "units": lanes,
    }


def ports(net="/sys/class/net", ib="/sys/class/infiniband"):
    """The ConnectX-7 side: the mlx5 network ports and the RDMA devices, as the kernel shows them."""
    out = {"cards": [], "ports": [], "rdma": []}
    for ln in run(["lspci", "-nn"]).splitlines():
        if "[15b3:" in ln or "Mellanox" in ln:
            out["cards"].append(re.sub(r"^\S+\s+", "", ln).strip())
    for dev in sorted(Path(net).glob("*")):
        driver = os.path.basename(os.path.realpath(dev / "device" / "driver")) if (dev / "device").exists() else ""
        if not driver.startswith("mlx5"):
            continue
        addrs = run(["ip", "-o", "-4", "addr", "show", "dev", dev.name])
        kinds = sorted({ip_kind(m) for m in re.findall(r"inet (\S+)", addrs)})
        out["ports"].append({"name": dev.name, "state": read(dev / "operstate"), "speed_mbps": read(dev / "speed"),
                             "mtu": read(dev / "mtu"), "ipv4": kinds})
    for port in sorted(Path(ib).glob("*/ports/*")):
        out["rdma"].append({"device": f"{port.parent.parent.name}/{port.name}", "state": read(port / "state"),
                            "rate": read(port / "rate"), "link": read(port / "link_layer")})
    return out


def software():
    return {
        "docker": run(["docker", "version", "--format", "{{.Server.Version}}"]).strip(),
        "nvidia_ctk": (run(["nvidia-ctk", "--version"]).splitlines() or [""])[0],
        "time_synced": run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"]).strip(),
        "free_gib": {"home": _free(HOME), "hf_cache": _free(Path(HOME) / ".cache" / "huggingface")},
    }


def _free(path):
    try:
        st = os.statvfs(os.path.realpath(path))
        return round(st.f_bavail * st.f_frsize / 2**30, 1)
    except OSError:
        return None


def peer(addr):
    """How the other box answers: the port the route takes, the loss and the round trip."""
    route = re.search(r"\bdev (\S+)", run(["ip", "route", "get", addr]))
    ping = run(["ping", "-c", "4", "-W", "1", addr], timeout=12)
    loss = re.search(r"(\d+(?:\.\d+)?)% packet loss", ping)
    rtt = re.search(r"= [\d.]+/([\d.]+)/", ping)
    return {"kind": ip_kind(addr), "via": route.group(1) if route else "",
            "loss_pct": float(loss.group(1)) if loss else None, "rtt_ms": float(rtt.group(1)) if rtt else None}


def report(facts):
    lines = ["dgx-spark-qwen38 pair probe (read-only, #45)"]
    for section, values in facts.items():
        lines.append(f"[{section}]")
        for k, v in values.items():
            lines.append(f"  {k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    return masked("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only facts a pair of Spark boxes needs (#45).")
    ap.add_argument("--peer", help="the other box's address on the ConnectX-7 link")
    ap.add_argument("--json", action="store_true", help="print the facts as JSON")
    args = ap.parse_args(argv)
    facts = {"box": box(), "repo": repo(), "connectx7": ports(), "software": software()}
    if args.peer:
        facts["peer"] = peer(args.peer)
    print(masked(json.dumps(facts, indent=2)) if args.json else report(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
