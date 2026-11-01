#!/usr/bin/env python3
"""get.sh's guard logic, offline, against a real git fixture.

get.sh is what every user runs, and nothing exercised its behaviour: shellcheck
and `bash -n` only prove it parses. The fixture builds a bare repo whose path
contains this project's slug (so get.sh's origin check passes), clones it, and
replaces install.sh with a stub that records its arguments, so the whole script
runs to its exec without touching a real install."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GET = os.path.join(REPO_DIR, "get.sh")
STUB = '#!/usr/bin/env bash\necho "INSTALL_RAN args=$*"\n'


# git here never inherits a GIT_DIR, GIT_WORK_TREE or GIT_INDEX_FILE (a git hook exports them), which would turn
# the fixture's commands onto that repository (code review, 2026-11-01)
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(*a, cwd, check=True):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=check, env=CLEAN)


def fixture(base):
    """A bare 'origin' whose URL carries the slug, plus a clone one commit behind."""
    origin = os.path.join(base, "hasso5703", "dgx-spark-qwen38.git")
    os.makedirs(origin)
    git("init", "-q", "--bare", "-b", "main", ".", cwd=origin)
    work = os.path.join(base, "work")
    git("clone", "-q", origin, work, cwd=base)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        git("config", k, v, cwd=work)
    with open(os.path.join(work, "install.sh"), "w") as f:
        f.write(STUB)
    os.chmod(os.path.join(work, "install.sh"), 0o755)
    git("add", "-A", cwd=work); git("commit", "-qm", "v1", cwd=work)
    git("push", "-q", "origin", "main", cwd=work)
    return origin, work


def isolated(base, extra=None):
    """What every run of get.sh here gets: PATH and nothing else of this shell's environment (no DIR, FORCE_UPDATE,
    SUDO_USER or GIT_* inherited), English messages, the fixture's folder for HOME, git with no network protocol
    but local paths, and no repository looked for above that folder (a TMPDIR inside a real clone would hand Case 1
    that clone). So no run reaches the network, a real clone of this repo or a real installer (code review,
    2026-11-01)."""
    return {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "HOME": base, "GIT_ALLOW_PROTOCOL": "file",
            "GIT_CEILING_DIRECTORIES": os.path.dirname(base), **(extra or {})}


def run_get(work, env_extra=None, args=()):
    here = os.path.dirname(os.path.abspath(work))
    return subprocess.run(["bash", GET, *args], cwd=here, capture_output=True, text=True,
                          env=isolated(here, env_extra) | {"DIR": work})


def main() -> None:
    base = tempfile.mkdtemp()
    try:
        _, work = fixture(base)

        # 1. Clean clone on main: get.sh runs through to the installer.
        r = run_get(work)
        assert r.returncode == 0, f"clean clone refused: {r.stderr}"
        assert "INSTALL_RAN" in r.stdout, r.stdout

        # 2. Arguments reach install.sh unchanged.
        r = run_get(work, args=["--no-service", "--no-start"])
        assert "INSTALL_RAN args=--no-service --no-start" in r.stdout, r.stdout

        # 3. An untracked file must NOT block, must survive, and must be reported.
        #    It cannot make the checkout stale and `reset --hard` leaves it alone;
        #    blocking on it used to fail the install of anyone keeping notes there.
        note = os.path.join(work, "my-notes.txt")
        with open(note, "w") as f:
            f.write("keep me")
        r = run_get(work)
        assert r.returncode == 0, f"an untracked file blocked the update: {r.stderr}"
        assert os.path.exists(note), "the untracked file was removed"
        assert open(note).read() == "keep me", "the untracked file was altered"
        assert "Untracked files kept in place" in r.stdout, r.stdout
        os.remove(note)

        # 4. A modified tracked file DOES block, and says so.
        with open(os.path.join(work, "install.sh"), "a") as f:
            f.write("# local edit\n")
        r = run_get(work)
        assert r.returncode != 0, "a modified tracked file must block"
        assert "modified tracked files" in r.stderr, r.stderr

        # 5. FORCE_UPDATE=1 stashes the tracked change and proceeds.
        r = run_get(work, env_extra={"FORCE_UPDATE": "1"})
        assert r.returncode == 0, f"FORCE_UPDATE did not recover: {r.stderr}"
        assert "INSTALL_RAN" in r.stdout, r.stdout
        assert git("stash", "list", cwd=work).stdout.strip(), "the change was not stashed"

        # 6. A clone that is not this repo is refused, whatever it contains.
        other = os.path.join(base, "other")
        git("clone", "-q", work, other, cwd=base)
        git("remote", "set-url", "origin", "https://github.com/someone/else.git", cwd=other)
        r = run_get(other)
        assert r.returncode != 0 and "not a clone of this repo" in r.stderr, r.stderr

        # 7. A side branch is refused without FORCE_UPDATE, accepted with it.
        git("checkout", "-qb", "side", cwd=work)
        r = run_get(work)
        assert r.returncode != 0 and "not 'main'" in r.stderr, r.stderr
        r = run_get(work, env_extra={"FORCE_UPDATE": "1"})
        assert r.returncode == 0, f"FORCE_UPDATE did not switch back to main: {r.stderr}"

        # 8. A release tagged after the clone reaches it: the cockpit's release check and
        #    `git describe` read the tags, and a fetch of main alone brought none, so a box
        #    updated by the one-liner kept naming the release it was cloned at (2026-10-01).
        git("checkout", "-q", "main", cwd=work)
        git("tag", "-a", "v9.9.9", "-m", "a release", cwd=work)
        git("push", "-q", "origin", "v9.9.9", cwd=work)
        git("tag", "-d", "v9.9.9", cwd=work)
        r = run_get(work)
        assert r.returncode == 0, f"the update failed: {r.stderr}"
        assert git("tag", "-l", "v9.9.9", cwd=work).stdout.strip() == "v9.9.9", "the release tag did not come"
        assert git("describe", "--tags", "--abbrev=0", cwd=work).stdout.strip() == "v9.9.9"

        # 9. A tag of the same name made here, on another commit, is kept, and the update
        #    goes on rather than failing on it.
        git("tag", "-a", "v9.9.8", "-m", "upstream", cwd=work)
        git("push", "-q", "origin", "v9.9.8", cwd=work)
        git("tag", "-d", "v9.9.8", cwd=work)
        git("commit", "-q", "--allow-empty", "-m", "local", cwd=work)
        git("tag", "v9.9.8", cwd=work)
        git("reset", "-q", "--hard", "HEAD~1", cwd=work)
        r = run_get(work)
        assert r.returncode == 0, f"a local tag of the same name failed the update: {r.stderr}"
        assert "INSTALL_RAN" in r.stdout, r.stdout

        # 10. The repository's new name (hasso5703/alzeo, from 2026-11-01) is this repo too, the old
        #     name stays accepted (every box installed before carries it), and another account's
        #     repository of the same name is not.
        for slug, ok in (("hasso5703/alzeo.git", True), ("someone/alzeo.git", False),
                         ("hasso5703/alzeo-site.git", False), ("xhasso5703/alzeo.git", False)):
            renamed = os.path.join(base, *slug.split("/"))
            os.makedirs(os.path.dirname(renamed), exist_ok=True)
            git("clone", "-q", "--bare", os.path.join(base, "hasso5703", "dgx-spark-qwen38.git"), renamed, cwd=base)
            clone = os.path.join(base, "clone-" + slug.replace("/", "-"))
            git("clone", "-q", renamed, clone, cwd=base)
            r = run_get(clone)
            if ok:
                assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"{slug} refused: {r.stderr}"
            else:
                assert r.returncode != 0 and "not a clone of this repo" in r.stderr, f"{slug} accepted: {r.stdout}"
            # and from inside the clone, with no DIR: its own origin picks it, as for the old name. Only
            # for the accepted one: a refused clone sends get.sh on to its default folder, which it would
            # clone from GitHub, and git refuses the network here at once
            if ok:
                r = subprocess.run(["bash", GET], cwd=clone, capture_output=True, text=True, env=isolated(base))
                assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"{slug} from inside: {r.stderr[-300:]}"

        # 11. Every form an origin of either name takes, HTTPS or SSH, with or without .git, as git keeps
        #     it: accepted, or refused as another repository. git refuses the network at once, after
        #     get.sh has said whether it takes the clone ("── Repo:" comes first). The old name keeps the
        #     pattern it always had, now whatever the case: every box installed before carries it.
        forms = {"https://github.com/hasso5703/alzeo": True, "https://github.com/hasso5703/alzeo.git": True,
                 "https://github.com/hasso5703/alzeo/": True, "git@github.com:hasso5703/alzeo.git": True,
                 "ssh://git@github.com/hasso5703/alzeo.git": True, "https://token@github.com/hasso5703/alzeo": True,
                 "https://github.com/hasso5703/dgx-spark-qwen38": True,
                 "git@github.com:hasso5703/dgx-spark-qwen38.git": True,
                 "https://github.com/hasso5703/alzeo.git/": True, "git@github.com:hasso5703/alzeo": True,
                 "ssh://git@github.com:22/hasso5703/alzeo": True, "https://user:pass@github.com/hasso5703/alzeo.git": True,
                 # GitHub's names ignore case: a clone typed with the brand's capital is this repo too
                 "https://github.com/Hasso5703/Alzeo": True, "https://github.com/hasso5703/ALZEO.git": True,
                 # a local clone given as its .git folder, as git keeps the path typed
                 "/srv/hasso5703/alzeo/.git": True, "/srv/hasso5703/alzeo/.git/": True,
                 # the old name as older boxes may carry it: capitals, a final slash, a .git folder, a port
                 "https://github.com/Hasso5703/DGX-Spark-Qwen38": True,
                 "https://github.com/hasso5703/dgx-spark-qwen38.git/": True,
                 "/srv/hasso5703/dgx-spark-qwen38/.git": True,
                 "https://token@github.com/hasso5703/dgx-spark-qwen38.git": True,
                 "ssh://git@github.com:22/hasso5703/dgx-spark-qwen38.git": True,
                 "https://github.com/hasso5703/alzeo-site": False, "https://github.com/xhasso5703/alzeo": False,
                 "https://github.com/someone/alzeo": False, "https://github.com/someone/dgx-spark-qwen38": False,
                 "https://github.com/Hasso5703/Alzeo-Site": False}
        def state(clone):
            """What a refused clone must keep: its origin, its commit, its files."""
            return (git("remote", "get-url", "origin", cwd=clone).stdout, git("rev-parse", "HEAD", cwd=clone).stdout,
                    git("status", "--porcelain", "--untracked-files=all", cwd=clone).stdout, sorted(os.listdir(clone)))
        for url, ok in forms.items():
            clone = os.path.join(base, "form-%d" % len(os.listdir(base)))
            git("clone", "-q", work, clone, cwd=base)
            git("remote", "set-url", "origin", url, cwd=clone)
            before = state(clone)
            r = run_get(clone)
            said = r.stdout + r.stderr
            assert r.returncode != 0, f"{url}: the network was refused, so get.sh cannot have gone through"
            if ok:
                assert "── Repo: " in said and "not a clone of this repo" not in said, f"{url} refused: {said[-300:]}"
            else:
                assert "not a clone of this repo" in said and "── Repo: " not in said, f"{url} taken: {said[-300:]}"
                assert state(clone) == before, f"{url}: the refused clone was changed"
        # a refusal shows the origin without the user name, password or token a URL can carry, which an
        # error pasted into an issue would publish: a password holding an @ or a / of its own, the user
        # of an scp-like address and a remote helper's URL included (the part after the first @, or the
        # whole address, was shown)
        for url, secret, shown in (
                ("https://someone:ghp_SECRET123@github.com/someone/fork.git", "ghp_SECRET123", "https://github.com/someone/fork.git"),
                ("https://someone:p@ss@github.com/someone/fork.git", "ss@", "https://github.com/someone/fork.git"),
                ("https://someone:pa/ss@github.com/someone/fork.git", "pa/ss", "https://github.com/someone/fork.git"),
                ("ghp_TOKEN9@github.com:someone/fork.git", "ghp_TOKEN9", "github.com:someone/fork.git"),
                ("gcrypt::https://someone:ghp_LEAK1@github.com/someone/private.git", "ghp_LEAK1",
                 "gcrypt::https://github.com/someone/private.git"),
                # forms sed alone left whole: a helper around an scp-like address, two helpers, a space before
                # the scheme (code review, 2026-11-01)
                ("gcrypt::ghp_TOK7@github.com:someone/private.git", "ghp_TOK7", "github.com:someone/private.git"),
                ("a::b::https://u:ghp_NEST@github.com/someone/fork.git", "ghp_NEST", "github.com/someone/fork.git"),
                ("  https://u:ghp_LEAD@github.com/someone/fork.git", "ghp_LEAD", "github.com/someone/fork.git")):
            clone = os.path.join(base, "form-%d" % len(os.listdir(base)))
            git("clone", "-q", work, clone, cwd=base)
            git("remote", "set-url", "origin", url, cwd=clone)
            r = run_get(clone)
            assert secret not in r.stdout + r.stderr, f"{secret} was printed: {r.stderr[-300:]}"
            assert f"(origin: {shown})" in r.stderr, r.stderr[-300:]
        # and a password holding a byte that is not UTF-8, under a UTF-8 locale, where sed's . does not
        # match it: the mask runs in the C locale. Without C.UTF-8 on this machine the check would prove nothing.
        locales = subprocess.run(["locale", "-a"], capture_output=True, text=True).stdout.lower().split()
        assert "c.utf8" in locales or "c.utf-8" in locales, "no C.UTF-8 locale here: the check below would prove nothing"
        clone = os.path.join(base, "form-%d" % len(os.listdir(base)))
        git("clone", "-q", work, clone, cwd=base)
        subprocess.run(["git", "remote", "set-url", "origin", b"https://someone:p\xe4ss@github.com/someone/fork.git"],
                       cwd=clone, check=True)
        r = subprocess.run(["bash", GET], capture_output=True,
                           env=isolated(base, {"LC_ALL": "C.UTF-8"}) | {"DIR": clone})
        assert b"p\xe4ss" not in r.stdout + r.stderr, r.stderr[-300:]
        assert b"(origin: https://github.com/someone/fork.git)" in r.stderr, r.stderr[-300:]

        # 12. The same rule from inside a clone, with no DIR (the one-liner run from the clone's folder): a
        #     clone of this repo under either name is the one used, any other repository is left alone and
        #     the default folder taken instead (cloned from REPO_URL, which git refuses here at once, and
        #     removes what it began: only get.sh's own words can tell, code review, 2026-11-01).
        for url, ok in forms.items():
            clone = os.path.join(base, "inside-%d" % len(os.listdir(base)))
            git("clone", "-q", work, clone, cwd=base)
            git("remote", "set-url", "origin", url, cwd=clone)
            before = state(clone)
            r = subprocess.run(["bash", GET], cwd=clone, capture_output=True, text=True, env=isolated(base))
            said = r.stdout + r.stderr
            assert r.returncode != 0, f"{url}: the network was refused, so get.sh cannot have gone through"
            # get.sh names the clone as git rev-parse resolves it (a TMPDIR behind a symlink, as on macOS)
            here = os.path.realpath(clone)
            if ok:
                assert f"── Repo: {here}" in said and "── Cloning" not in said, f"{url} from inside: not used: {said[-300:]}"
            else:
                assert f"── Repo: {here}" not in said and "── Cloning" in said, f"{url} from inside: {said[-300:]}"
                assert state(clone) == before, f"{url} from inside: the clone left alone was changed"

        # 13. get.sh stays within bash 3.2, macOS's /bin/bash: there a ${x,,} is a bad substitution that
        #     stops the script, after it may have cloned the default folder (code review, 2026-11-01).
        code = "\n".join(l for l in open(GET, encoding="utf-8").read().splitlines() if not l.lstrip().startswith("#"))
        for construct in (",,}", "^^}", "declare -A", "declare -l", "declare -u", "typeset -A", "typeset -l",
                          "typeset -u", "mapfile", "readarray", "coproc", "|&", ";;&", ";&", "&>>", "globstar",
                          "lastpipe"):
            assert construct not in code, f"get.sh uses {construct!r}, which bash 3.2 does not have"
        assert not re.search(r"\$\{[A-Za-z_0-9]+[,^]\}", code), "get.sh uses a ${x,} or ${x^}, which bash 3.2 does not have"
        assert not re.search(r"\$\{[^}]*@[A-Za-z]\}", code), "get.sh uses a ${x@L} or the like, which bash 3.2 does not have"

        # 14. Many untracked files, or many modified ones: under set -o pipefail a `| head` closed the pipe early,
        #     its writer took SIGPIPE and get.sh ended with 141, silently, before the update or before its advice
        #     (code review, 2026-11-01; a .venv-mirror/ in the clone, as MIRROR.md makes one, holds thousands).
        #     A clone holding 3000 untracked files is updated, and one with 600 modified tracked files is refused
        #     with the whole advice.
        tail = "-with-a-name-long-enough-that-the-list-outgrows-the-pipe-buffer-of-the-kernel.py"
        bare = os.path.join(base, "hasso5703", "dgx-spark-qwen38.git")
        many = os.path.join(base, "many")
        git("clone", "-q", bare, many, cwd=base)
        os.makedirs(os.path.join(many, "venv-like"))
        for i in range(3000):
            open(os.path.join(many, "venv-like", f"f{i:05d}{tail}"), "w").close()
        r = run_get(many)
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"3000 untracked files: rc {r.returncode}: {r.stdout[-200:]}"
        assert "Untracked files kept in place" in r.stdout and "── At:" in r.stdout, r.stdout[-300:]
        mod = os.path.join(base, "mod")
        git("clone", "-q", bare, mod, cwd=base)
        for k, v in (("user.email", "t@t"), ("user.name", "t")):
            git("config", k, v, cwd=mod)
        os.makedirs(os.path.join(mod, "tracked"))
        for i in range(600):
            open(os.path.join(mod, "tracked", f"f{i:04d}{tail}"), "w").close()
        git("add", "-A", cwd=mod); git("commit", "-qm", "600 files", cwd=mod)
        for i in range(600):
            with open(os.path.join(mod, "tracked", f"f{i:04d}{tail}"), "w") as f:
                f.write("changed")
        r = run_get(mod)
        assert r.returncode == 1, f"600 modified files: rc {r.returncode} (141 = cut by SIGPIPE): {r.stderr[-200:]}"
        assert "modified tracked files" in r.stderr and "Fix with ONE of" in r.stderr and "FORCE_UPDATE=1" in r.stderr, r.stderr[-300:]

        print("test_get_sh: OK")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
