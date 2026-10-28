#!/usr/bin/env python3
"""What a pair of DGX Spark (or GB10) boxes needs to know about each one, read-only (#45).

Run it on each box, no sudo needed, and paste what it prints:

    python3 ~/dgx-spark-qwen38/pair-probe.py                      # this box
    python3 ~/dgx-spark-qwen38/pair-probe.py --peer 192.168.100.2  # and the link to the other one

It reads the OS, the kernel, the GPU driver, this repo's installed release and lanes, the ConnectX-7
ports, their RDMA side and the mlx5 modules, Docker, the time sync and the free disk (the home and the
model cache the lanes mount), and, given --peer (an address or a name), how the other box answers over
which port. It changes nothing and starts nothing.

What it never prints: an API key or any file that holds one (it opens none), a MAC address, an IP
address (only its kind: private, link-local, shared, public), the host's name, your home directory
or user name. --json prints the same facts for a program.
"""
import argparse
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

HOME = str(Path.home())
HERE = Path(__file__).resolve().parent
CFG = Path(HOME) / ".config" / "qwen38"
UNITS = Path("/etc/systemd/system")


def run(argv, timeout=8.0, any_exit=False):
    """A command's output, or "" when it is missing, fails or takes too long. Never a shell. any_exit
    keeps what a command that ends non-zero printed: ping ends 1 when nothing answered, and says so."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 or any_exit else ""
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
    addresses of either version (their kind only), the home directory, and the host's name as a whole
    word (a name like "spark" must not eat "dgx-spark-qwen38")."""
    host = socket.gethostname() if host is None else host
    text = re.sub(r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b", "<mac>", text, flags=re.I)
    def ip(m):
        kind = ip_kind(m.group(0))
        return m.group(0) if kind == "unknown" else f"<{kind}>"      # a version like 580.178.04.1 stays
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?\b", ip, text)
    def ip6(m):
        addr = m.group(1).split("%")[0]
        try:
            ipaddress.IPv6Address(addr)
        except ValueError:
            return m.group(0)                                          # a time like 12:34:56 stays
        return f"<{ip_kind(addr)}>"
    text = re.sub(r"(?<![\w:.])((?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?:%[\w.-]+)?)(?:/\d{1,3})?(?![\w:])",
                  ip6, text, flags=re.I)
    text = text.replace(HOME, "~")
    if host:
        text = re.sub(r"(?<![\w.-])" + re.escape(host) + r"(?![\w-])", "<host>", text)
    return text


def masked_facts(facts, host=None):
    """The facts with each value masked, never their names: a host named "gpu" or "home" renamed the
    keys of --json and the labels of the report when the printed text was masked whole (review,
    2026-10-28)."""
    host = socket.gethostname() if host is None else host
    if isinstance(facts, dict):
        return {k: masked_facts(v, host) for k, v in facts.items()}
    if isinstance(facts, list):
        return [masked_facts(v, host) for v in facts]
    return masked(facts, host) if isinstance(facts, str) else facts


def box(release="/etc/dgx-release"):
    # DGX OS appends a DGX_OTA_VERSION line at each update: the last one is the version it runs, not
    # DGX_SWBUILD_VERSION, the image it was installed from. DGX_SERIAL_NUMBER is in the same file, never
    # shown: only the keys below are read out.
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
    if not isinstance(inst, dict):      # a partial write or a hand edit: valid JSON, not the record
        inst = {}
    field = lambda k: inst.get(k) if isinstance(inst.get(k), str) else ""
    # the clone the box was installed from (get.sh's DIR= puts it anywhere), else the one this file is in
    where = field("repo_dir") if field("repo_dir") and Path(field("repo_dir")).is_dir() else str(HERE)
    git = lambda *a: run(["git", "-C", where, *a]).strip()
    # every installed unit, loaded or not: systemd drops an inactive unit nothing refers to from memory,
    # and list-units no longer shows it (review, 2026-10-28)
    states = {}
    for ln in run(["systemctl", "list-units", "--all", "--no-legend", "--plain", "--type=service",
                   "qwen38-*"]).splitlines():
        parts = ln.split()
        if len(parts) >= 4 and parts[0].endswith(".service"):
            states[parts[0]] = f"{parts[2]}/{parts[3]}"
    lanes = {}
    for ln in run(["systemctl", "list-unit-files", "--no-legend", "--plain", "--type=service",
                   "qwen38-*"]).splitlines():
        parts = ln.split()
        if len(parts) >= 2 and parts[0].endswith(".service"):
            lanes[parts[0][:-8]] = states.get(parts[0], "inactive/dead") + (" (boot)" if parts[1] == "enabled" else "")
    return {
        "release": field("release"),
        "commit": field("commit")[:7],
        "lane": field("lane"),
        "checkout": git("rev-parse", "--short", "HEAD"),
        "branch": git("symbolic-ref", "-q", "--short", "HEAD"),
        "units": lanes,
    }


def ports(net="/sys/class/net", ib="/sys/class/infiniband", modules="/sys/module"):
    """The ConnectX-7 side: the mlx5 network ports and the RDMA devices, as the kernel shows them, the
    mlx5 modules it has loaded, and whether lspci is there to look: on the reference GX10 the modules
    are loaded and lspci shows no card, which empty lists alone could not tell from no lspci."""
    out = {"cards": [], "ports": [], "rdma": [],
           "mlx5_modules": [m for m in ("mlx5_core", "mlx5_ib") if (Path(modules) / m).is_dir()],
           "lspci": "found" if shutil.which("lspci") else "missing"}
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


def software(units=UNITS):
    return {
        "docker": run(["docker", "version", "--format", "{{.Server.Version}}"]).strip(),
        "nvidia_ctk": (run(["nvidia-ctk", "--version"]).splitlines() or [""])[0],
        "time_synced": run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"]).strip(),
        "free_gib": {"home": _free(HOME), "hf_cache": _free(hf_cache(units))},
    }


def hf_cache(units=UNITS):
    """Where the lanes keep their weights: the cache the 27B unit mounts, which is install.sh's HF_CACHE=
    and is kept across updates, else the default one."""
    m = re.search(r"-v ([^ :]+):/root/\.cache/huggingface\b", read(Path(units) / "qwen38-sglang.service"))
    return re.sub("/+", "/", m.group(1)) if m else str(Path(HOME) / ".cache" / "huggingface")


def _free(path):
    try:
        st = os.statvfs(os.path.realpath(path))
        return round(st.f_bavail * st.f_frsize / 2**30, 1)
    except OSError:
        return None


def peer(name):
    """How the other box answers: the port the route takes, the loss and the round trip. A name is
    resolved first, as ping would, since ip route get takes only an address, and what starts with a dash
    is no address: ping would take it for an option."""
    out = {"kind": "unknown", "via": "", "loss_pct": None, "rtt_ms": None}
    if name.startswith("-"):
        return {**out, "error": "not an address"}
    try:
        addr = socket.getaddrinfo(name, None, proto=socket.IPPROTO_TCP)[0][4][0]
    except (OSError, UnicodeError, IndexError):
        return {**out, "error": "the name does not resolve"}
    route = re.search(r"\bdev (\S+)", run(["ip", "route", "get", addr]))
    # ping ends 1 when nothing answered: the link that is down is the one this check is for
    ping = run(["ping", "-c", "4", "-W", "1", addr], timeout=12, any_exit=True)
    loss = re.search(r"(\d+(?:\.\d+)?)% packet loss", ping)
    rtt = re.search(r"= [\d.]+/([\d.]+)/", ping)
    return {"kind": ip_kind(addr), "via": route.group(1) if route else "",
            "loss_pct": float(loss.group(1)) if loss else None, "rtt_ms": float(rtt.group(1)) if rtt else None}


def report(facts):
    lines = ["dgx-spark-qwen38 pair probe (read-only, #45)"]
    for section, values in masked_facts(facts).items():
        lines.append(f"[{section}]")
        for k, v in values.items():
            lines.append(f"  {k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only facts a pair of Spark boxes needs (#45).")
    ap.add_argument("--peer", help="the other box's address on the ConnectX-7 link")
    ap.add_argument("--json", action="store_true", help="print the facts as JSON")
    args = ap.parse_args(argv)
    facts = {"box": box(), "repo": repo(), "connectx7": ports(), "software": software()}
    if args.peer:
        facts["peer"] = peer(args.peer)
    print(json.dumps(masked_facts(facts), indent=2) if args.json else report(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
