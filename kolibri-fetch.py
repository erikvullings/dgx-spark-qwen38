#!/usr/bin/env python3
"""The Kolibri lane's checkpoint, made whole in the cache: run inside the lane's own image by
install-kolibri.sh (step 3) and switch-model.sh (kolibri, kolibri-nvfp4), the one place it is
written. The two had their own copies, and they had drifted: only the switch checked that every
shard the index names is there (found in review, 2026-10-14).

The snapshot is first asked offline; a snapshot missing a shard, or none, is fetched, after a
check of the room the rest needs (KOL_FREE_GB, when given), with resumable retries and the byte
count hf-progress.py prints when it is mounted beside this file. Environment: KOL_MODEL, KOL_REV,
KOL_NEED_GB, KOL_FREE_GB (optional), QWEN38_REPO_DIR (where hf-progress.py is, optional), and the
usual HF_HOME, HF_TOKEN. Prints the snapshot's path on success.
"""
import contextlib
import importlib.util
import json
import os
import time
from huggingface_hub import constants, snapshot_download


def reporting(*args, **kwargs):
    """hf-progress.py's byte count while the rest comes; a progress line is never worth a download."""
    try:
        spec = importlib.util.spec_from_file_location(
            "hf_progress", os.path.join(os.environ.get("QWEN38_REPO_DIR")
                                        or os.path.dirname(os.path.abspath(__file__)), "hf-progress.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.reporting(*args, **kwargs)
    except Exception:
        return contextlib.nullcontext()


def complete(path):
    """Every shard the index names is in the snapshot (a file appears there only once its
    blob is whole), as install.sh's ckpt_cached reads it: huggingface_hub's own offline
    check sees a missing file only when it cached the repo's tree, which a download made
    with another tool or an older hub never wrote, and vLLM then dies at every start."""
    try:
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            shards = set(json.load(f)["weight_map"].values())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        shards = {"model.safetensors"}
    return bool(shards) and all(os.path.exists(os.path.join(path, s)) for s in shards)


repo, rev = os.environ["KOL_MODEL"], os.environ["KOL_REV"]
try:
    path = snapshot_download(repo, revision=rev, local_files_only=True)
    if not complete(path):
        raise FileNotFoundError(f"a shard the index of {repo} names is missing")
except Exception:
    # Refused before the first byte rather than stopped by a full disk halfway. What the
    # cache already holds of the repo comes off the need: a fetch that stopped resumes.
    blobs = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "blobs")
    have = sum(e.stat().st_size for e in os.scandir(blobs) if e.is_file()) if os.path.isdir(blobs) else 0
    free = int(os.environ.get("KOL_FREE_GB") or 0)
    need = max(int(os.environ["KOL_NEED_GB"]) - have // 10**9, 1)
    if free and free < need:
        raise SystemExit(f"{free} GB free in the cache, and {repo} needs about {need} GB more: free some space first")
    print("not complete in the cache, fetching the rest", flush=True)
    with reporting(repo, revision=rev):
        for attempt in range(1, 6):  # a new attempt keeps every file already whole
            try:
                path = snapshot_download(repo, revision=rev)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"download interrupted ({type(e).__name__}), resuming ({attempt}/5)...", flush=True)
                time.sleep(10)
print(path)
