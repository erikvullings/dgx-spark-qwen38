#!/usr/bin/env bash
# Run this repo's CI shell steps locally, exactly as GitHub does (bash -e), before
# pushing. A step is reported as SKIPPED when it calls a tool missing on this box, or
# when a package its runner-only `pip install` line provides is missing here.
# Usage: ./ci-local.sh [step-name-substring]
# The workflow's jobs run side by side, as on GitHub, each in a checkout of its own: a
# detached worktree of HEAD, so the tree has to be clean (what runs is HEAD^{tree}).
# Within a job the steps run in order. CI_LOCAL_SERIAL=1 runs every step in this checkout,
# one after the other, as before (a dirty tree included).
# Exit status: 1 when a step failed, 3 when nothing ran (a filter that matched no step,
# or only skipped ones), 0 otherwise.
set -u
FILTER="${1:-}"
python3 -u - "$FILTER" <<'PY'
import os, re, signal, subprocess, sys, yaml, shutil, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
flt = sys.argv[1]
serial = os.environ.get("CI_LOCAL_SERIAL") == "1"
say = threading.Lock()
# Each step's wall time, said on its line and summed up at the end: where a run's hour goes
# is the first thing to know before making it shorter (2026-10-24).
took, t_run = [], time.monotonic()
d = yaml.safe_load(open(".github/workflows/ci.yml"))
ok = fail = skip = 0
# A run stopped through its process group gets SIGTERM, which ended python without its
# finally blocks: the step's throwaway HOME, with the keys the cockpit tests write, and
# the stubs stayed in TMPDIR (found 2026-10-07). As an exception, it runs them.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
# A tool is missing when a step CALLS it, not when its name appears in the step's text.
# The test was a substring one: without gh, 13 of 61 steps were skipped for containing
# "through", "high" or "github", the sudoers gate among them, and docker is named by two
# steps that never run it (found in review, 2026-09-24). Each missing tool gets a stub,
# first on PATH, that records the call and fails the way a missing command does.
stubs = tempfile.mkdtemp(prefix="ci-local-stubs-")
for tool in ("shellcheck", "docker", "gh"):
    if not shutil.which(tool):
        with open(os.path.join(stubs, tool), "w") as f:
            f.write(f'#!/bin/sh\necho {tool} >> "$CI_LOCAL_CALLED"\n'
                    f'echo "ci-local: {tool} is not installed here" >&2\nexit 127\n')
        os.chmod(os.path.join(stubs, tool), 0o755)
PIP = re.compile(r"pip install\s+'([A-Za-z0-9_.-]+)==")


def have(pkg, env):
    """Whether this box has what a runner-only pip line installs, found the way the steps
    find it (tests/testpy.sh, under the step's own HOME)."""
    if pkg == "ruff":
        return shutil.which("ruff") is not None
    return subprocess.run(["./tests/testpy.sh", pkg.replace("-", "_")], env=env,
                          capture_output=True).returncode == 0


def run_job(job, steps, cwd):
    """The job's steps in order, in cwd; each line said as it ends. (ok, fail, skip)"""
    ok = fail = skip = 0
    tag = f"[{job}] " if not serial else ""
    for st in steps:
        run = st.get("run")
        if not run:
            continue
        name = st.get("name") or run.splitlines()[0][:60]
        if flt and flt not in name:
            continue
        # Run every step under a throwaway HOME. A GitHub runner has no
        # ~/.config/qwen38, and a test that quietly read the developer's own API
        # key passed here and failed there; local has to be the harder of the two.
        home = tempfile.mkdtemp(prefix="ci-local-home-")
        called = os.path.join(home, ".ci-local-called")
        # and a TMPDIR inside it, removed with it: the suite's own temporary directories
        # (and its scripts' mktemp) stayed in /tmp after every run, 761 of them on the
        # reference box on 2026-10-03, some with mtimes in 2027 that read as files changed
        # by whatever ran next
        tmp = os.path.join(home, "tmp")
        os.mkdir(tmp)
        env = {**os.environ, "HOME": home, "TMPDIR": tmp, "CI_LOCAL_CALLED": called,
               "PATH": stubs + os.pathsep + os.environ.get("PATH", "")}
        r, tools, dt = None, [], 0.0
        try:
            # The runner installs its pinned packages; here they have to be there
            # already. The line is dropped, and what it installs is looked for: the
            # old test looked for ruff alone, and without ruff it skipped all six
            # pip steps, coverage and hypothesis included. A line ending in
            # `|| echo ...` is one the step runs without, and says so.
            need = []
            for ln in (ln for ln in run.splitlines() if "pip install" in ln):
                m = PIP.search(ln)
                if m and "|| echo" not in ln and not have(m.group(1), env):
                    need.append(m.group(1))
            if need:
                with say:
                    print(f"  {tag}SKIP {name} (missing: {', '.join(need)})")
                skip += 1
                continue
            script = "\n".join(ln for ln in run.splitlines() if "pip install" not in ln)
            t0 = time.monotonic()
            r = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True,
                               cwd=cwd, env=env)
            dt = time.monotonic() - t0
            tools = sorted(set(open(called).read().split())) if os.path.exists(called) else []
        finally:
            shutil.rmtree(home, ignore_errors=True)
        with say:
            took.append((dt, f"{tag}{name}"))
            if tools:
                print(f"  {tag}SKIP {name} (missing: {', '.join(tools)})"); skip += 1
            elif r.returncode == 0:
                print(f"  {tag}ok   {name} ({dt:.0f} s)"); ok += 1
            else:
                print(f"  {tag}FAIL {name} (rc={r.returncode}, {dt:.0f} s)")
                print("       " + (r.stdout + r.stderr).strip().replace("\n", "\n       ")[-600:]); fail += 1
    return ok, fail, skip


jobs = [(n, j.get("steps", [])) for n, j in d["jobs"].items()]
trees, per_job = [], {}
try:
    if serial:
        for n, steps in jobs:
            o, f_, s_ = run_job(n, steps, os.getcwd()); ok += o; fail += f_; skip += s_
    else:
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        if dirty.strip():
            print("the tree is not clean: the jobs run in checkouts of HEAD, so commit first, or run "
                  "with CI_LOCAL_SERIAL=1 to test this checkout as it is:\n" + dirty)
            sys.exit(2)
        base = tempfile.mkdtemp(prefix="ci-local-jobs-")
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD^{tree}"], capture_output=True, text=True).stdout.strip()
        print(f"tree {head}: {len(jobs)} jobs side by side, each in a worktree of HEAD")

        def one(n_steps):
            n, steps = n_steps
            wt = os.path.join(base, n)
            subprocess.run(["git", "worktree", "add", "-q", "--detach", wt, "HEAD"], check=True)
            trees.append(wt)
            t0 = time.monotonic()
            res = run_job(n, steps, wt)
            per_job[n] = time.monotonic() - t0
            return res
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            for o, f_, s_ in pool.map(one, jobs):
                ok += o; fail += f_; skip += s_
finally:
    shutil.rmtree(stubs, ignore_errors=True)
    for wt in trees:
        subprocess.run(["git", "worktree", "remove", "--force", wt], capture_output=True)
    if trees:
        subprocess.run(["git", "worktree", "prune"], capture_output=True)
if per_job:
    print(f"\njobs, of {time.monotonic() - t_run:.0f} s in all (the longest sets the run):")
    for n, dt in sorted(per_job.items(), key=lambda x: -x[1]):
        print(f"  {dt:6.0f} s  {n}")
if took:
    print(f"\nslowest steps, of {time.monotonic() - t_run:.0f} s in all:")
    for dt, name in sorted(took, reverse=True)[:10]:
        print(f"  {dt:6.0f} s  {name}")
print(f"\nCI local: {ok} ok, {fail} echecs, {skip} sautes")
if fail:
    sys.exit(1)
if not ok:
    print("nothing ran" + (f": no step matches '{flt}'" if not skip else ": every step it matched was skipped"))
    sys.exit(3)
PY
