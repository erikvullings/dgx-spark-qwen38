#!/usr/bin/env python3
"""Where the Flash FP4-Local lane's engine reads its n-gram (PLE) table from, and the copy that puts
it there.

The table of nvidia/Qwen3.8-Flash-Next-FP4-Local (26.8 GiB) lives inside seven of the checkpoint's
21 files, and vLLM reads it in place at every token, rows at random over io_uring. Read from a USB
SSD (the reference box's cache, 2026-10-19), a request alone decoded 15.7 tokens a second and eight
together 15.6 in all; with the same seven files on the internal NVMe, 23.4 and 77.7, and the first
token of a 1,024-token prompt came twice as fast. The weights the engine loads once at boot can stay
where the cache is: only the table is read at every token. Hasan's choice (2026-10-20): the table is
read where the cache is when the cache is on the disk of the home directory (a new box's default),
and otherwise from copies of those files in a folder of the home directory, which the unit mounts
over the cache's own, file by file. The cache itself is never touched.

    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] plan     what goes where, in bytes
    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] copy     make the copies whole
    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] mounts   the unit's -v arguments, one line

--dir is FLASH_LOCAL_PLE_DIR: a folder for the copies whatever disk the cache is on, or "none" to
read the table where the cache is. Without it, HOME/flash-local-ple when the cache is on another
disk than HOME. A copy is made as NAME.part, checked against its blob's name (the sha256 of its
content, as the Hub's cache names every large file) and only then renamed: a copy cut short is
never mounted. A copy already there is checked the same way at every copy. copy removes from the
folder the copies no file of the table names any more (an earlier revision's), and nothing else.
Exit status 2: the checkpoint is not whole in the cache.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

BLOB = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_HUB = "/root/.cache/huggingface/hub"


def table_files(snapshot: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """(the file's path in the snapshot, its blob), for every file holding a tensor of the table."""
    index = snapshot / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    names = sorted({f for t, f in weight_map.items() if ".ngram_embedding." in t})
    if not names:
        raise SystemExit(f"{index} names no n-gram table: is this the FP4-Local checkpoint?")
    out = []
    for name in names:
        link = snapshot / name
        blob = link.resolve()
        if not blob.is_file() or not BLOB.match(blob.name):
            print(f"{link} is not in the cache as a blob: fetch the checkpoint first", file=sys.stderr)
            sys.exit(2)
        out.append((name, blob))
    return out


def where(args, blobs_dir: pathlib.Path) -> pathlib.Path | None:
    """The folder of the copies, or None: the table is read where the cache is."""
    if args.dir == "none":
        return None
    if args.dir:
        return pathlib.Path(args.dir)
    if os.stat(blobs_dir).st_dev == os.stat(args.home).st_dev:
        return None
    return pathlib.Path(args.home) / "flash-local-ple"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hf_cache")
    ap.add_argument("repo")
    ap.add_argument("rev")
    ap.add_argument("--home", required=True)
    ap.add_argument("--dir", default="")
    ap.add_argument("action", choices=("plan", "copy", "mounts"))
    args = ap.parse_args()
    repo_dir = pathlib.Path(args.hf_cache) / "hub" / ("models--" + args.repo.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / args.rev
    if not (snapshot / "model.safetensors.index.json").is_file():
        print(f"no snapshot of {args.repo} @ {args.rev[:12]} in {args.hf_cache}: fetch the checkpoint first",
              file=sys.stderr)
        sys.exit(2)
    files = table_files(snapshot)
    folder = where(args, repo_dir / "blobs")
    total = sum(b.stat().st_size for _, b in files)
    if args.action == "plan":
        if folder is None:
            print(f"in place: {len(files)} files, {total} bytes, read from {repo_dir / 'blobs'}")
            return
        missing = sum(b.stat().st_size for _, b in files if not (folder / b.name).is_file())
        print(f"copies: {len(files)} files, {total} bytes, {missing} to copy into {folder}")
        return
    if args.action == "mounts":
        if folder is None:
            return
        absent = [b.name for _, b in files if not (folder / b.name).is_file()]
        if absent:
            print(f"{len(absent)} copies are not in {folder}: run the copy first", file=sys.stderr)
            sys.exit(1)
        mount = f"{CONTAINER_HUB}/{repo_dir.name}/blobs"
        print(" ".join(f"-v {folder / b.name}:{mount}/{b.name}:ro" for _, b in files))
        return
    if folder is None:
        print(f"the table is read in place, from {repo_dir / 'blobs'}: no copy")
        return
    folder.mkdir(parents=True, exist_ok=True)
    keep = {b.name for _, b in files}
    for name, blob in files:
        dest = folder / blob.name
        # a copy already there is checked again (15 s for the seven on the reference box): one that
        # changed since would feed the model wrong embeddings without a single error
        if dest.is_file() and dest.stat().st_size == blob.stat().st_size and sha256(dest) == blob.name:
            continue
        part = folder / (blob.name + ".part")
        free = shutil.disk_usage(folder).free
        if free < blob.stat().st_size + (1 << 30):
            raise SystemExit(f"{free // 10**9} GB free under {folder}, {name} needs {blob.stat().st_size // 10**9 + 1} GB:"
                             " free some space, or set FLASH_LOCAL_PLE_DIR")
        print(f"copying {name} ({blob.stat().st_size / 10**9:.1f} GB) into {folder}", flush=True)
        shutil.copyfile(blob, part)
        if sha256(part) != blob.name:
            part.unlink()
            raise SystemExit(f"the copy of {name} does not match its sha256: the cache's file may be damaged"
                             " (fetch the checkpoint again), or the disk of the copy is failing")
        part.rename(dest)
    for p in folder.iterdir():
        stem = p.name[:-5] if p.name.endswith(".part") else p.name
        if BLOB.match(stem) and stem not in keep:
            p.unlink()
            print(f"removed {p.name[:12]}..., a copy no file of this table names any more")
    print(f"the table is read from {folder}: {len(files)} files, {total / 10**9:.1f} GB, checked")


if __name__ == "__main__":
    main()
