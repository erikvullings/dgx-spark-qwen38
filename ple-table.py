#!/usr/bin/env python3
"""Where the Flash FP4-Local lane's engine reads its n-gram (PLE) table from, and the copy that puts
it there.

The table of nvidia/Qwen3.8-Flash-Next-FP4-Local (26.8 GiB) lives inside seven of the checkpoint's
21 files, and vLLM reads it in place at every token, rows at random over io_uring. Read from a USB
SSD (the reference box's cache, 2026-10-19), a request alone got 15.7 tokens a second and eight
together 15.6 in all; with the same seven files on the internal NVMe, 23.4 and 77.7, and the first
token of a 1,024-token prompt came twice as fast (1.49 s against 0.75). The weights the engine loads once at boot can stay
where the cache is: only the table is read at every token. Hasan's choice (2026-10-20): the table is
read where the cache is when the cache is on the disk of the home directory (a new box's default),
and otherwise from copies of those files in a folder of the home directory, which the unit mounts
over the cache's own, file by file. The cache itself is never touched.

    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] plan     what goes where, in bytes
    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] copy     make the copies whole
    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] mounts   the unit's -v arguments, one line
    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] check    the folder refused or not, no checkpoint needed
    ple-table.py HF_CACHE REPO REV --home HOME [--dir DIR] --need-gb N room
                                                                   the copies' room on their disk, before any download

--dir is FLASH_LOCAL_PLE_DIR: a folder for the copies whatever disk the cache is on, or "none" to
read the table where the cache is, never a folder inside the cache, and its path one a unit can carry
(install.sh's rule). Without it, HOME/flash-local-ple
when the cache is on another disk than HOME. A copy is made as NAME.part, checked against its blob's
name (the sha256 of its content, as the Hub's cache names every large file) and only then renamed: a
copy cut short is never mounted. A copy already there is checked the same way at every copy. The
folder's .qwen38-ple-copies lists the copies this tool keeps there, and copy removes those of them no
file of the table names any more (an earlier revision's), and nothing else: a file named like a blob
that it did not make stays, whatever its name (found in review, 2026-10-20: it removed every such
file, the blobs of a model included). Exit status 2: the checkpoint is not whole in the cache.
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
# the copies this tool keeps in the folder, one name a line: the only files it ever removes there
OURS = ".qwen38-ple-copies"
# install.sh's rule for every path a unit carries: in a docker -v inside a quoted bash -c line, and in
# systemd's own syntax, a space, a colon, a quote, a $ or a % broke the unit it went into
PLAIN = re.compile(r"/[A-Za-z0-9._/+@-]*")


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
        folder = pathlib.Path(args.dir)
        if not PLAIN.fullmatch(args.dir):
            raise SystemExit(f"FLASH_LOCAL_PLE_DIR={args.dir}: the lane's unit cannot carry that path. Use an absolute"
                             " path of letters, digits and . _ / + @ - only, as for HF_CACHE.")
        # inside the cache, the copies would be the cache's own files (its blobs folder), or files the
        # Hub's tools take for the cache's
        cache = pathlib.Path(args.hf_cache).resolve()
        if folder.resolve() == cache or cache in folder.resolve().parents:
            raise SystemExit(f"FLASH_LOCAL_PLE_DIR={args.dir} is inside the Hugging Face cache ({args.hf_cache}):"
                             " the copies go in a folder outside it, or FLASH_LOCAL_PLE_DIR=none reads the table"
                             " where the cache is")
        return folder
    if os.stat(blobs_dir).st_dev == os.stat(args.home).st_dev:
        return None
    return pathlib.Path(args.home) / "flash-local-ple"


def kept(folder: pathlib.Path) -> set[str]:
    """The names of the copies this tool keeps in the folder."""
    try:
        return {n for n in (folder / OURS).read_text().split() if BLOB.match(n)}
    except OSError:
        return set()


def keep_list(folder: pathlib.Path, names: set[str]) -> None:
    part = folder / (OURS + ".part")
    part.write_text("".join(n + "\n" for n in sorted(names)))
    part.rename(folder / OURS)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def existing(path: pathlib.Path) -> pathlib.Path:
    while not path.exists():
        path = path.parent
    return path


def room(args, repo_dir: pathlib.Path) -> None:
    """The room the copies need on their folder's disk, said before anything is downloaded: the copy
    refused only after the checkpoint's 103 GB were fetched (found in review, 2026-10-20). With the
    checkpoint in the cache, the copies missing; before it, --need-gb, the table's size."""
    folder = where(args, existing(repo_dir / "blobs"))
    if folder is None:
        print("the table is read where the cache is: no copy to make room for")
        return
    need = args.need_gb * 10 ** 9
    snapshot = repo_dir / "snapshots" / args.rev
    if (snapshot / "model.safetensors.index.json").is_file():
        try:
            need = sum(b.stat().st_size for _, b in table_files(snapshot)
                       if not ((folder / b.name).is_file() and (folder / b.name).stat().st_size == b.stat().st_size))
        except SystemExit:
            pass              # not whole yet: the table's size, as before the download
    free = shutil.disk_usage(existing(folder)).free
    if free < need + (1 << 30):
        raise SystemExit(f"{free // 10 ** 9} GB free under {folder}, and the copies of the n-gram table need about"
                         f" {need // 10 ** 9 + 1} GB there: free some space, set FLASH_LOCAL_PLE_DIR to a folder on a"
                         " disk with the room, or FLASH_LOCAL_PLE_DIR=none to read the table where the cache is"
                         " (slower, docs/external-disk.md)")
    print(f"room for the table's copies under {folder}: {free // 10 ** 9} GB free, about {need // 10 ** 9 + 1} GB needed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hf_cache")
    ap.add_argument("repo")
    ap.add_argument("rev")
    ap.add_argument("--home", required=True)
    ap.add_argument("--dir", default="")
    ap.add_argument("--need-gb", type=int, default=0)
    ap.add_argument("action", choices=("plan", "copy", "mounts", "check", "room"))
    args = ap.parse_args()
    repo_dir = pathlib.Path(args.hf_cache) / "hub" / ("models--" + args.repo.replace("/", "--"))
    if args.action == "check":
        # the folder chosen, refused before the installer keeps it for the runs after; no checkpoint needed
        if args.dir and args.dir != "none":
            where(args, repo_dir / "blobs")
        return
    if args.action == "room":
        room(args, repo_dir)
        return
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
    ours = kept(folder)
    for name, blob in files:
        dest = folder / blob.name
        # listed before it is written: a copy cut short is this tool's to finish or remove
        ours.add(blob.name)
        keep_list(folder, ours)
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
    for stale in sorted(ours - keep):
        for p in (folder / stale, folder / (stale + ".part")):
            if p.is_file():
                p.unlink()
                print(f"removed {p.name[:12]}..., a copy no file of this table names any more")
    keep_list(folder, keep)
    print(f"the table is read from {folder}: {len(files)} files, {total / 10**9:.1f} GB, checked")


if __name__ == "__main__":
    main()
