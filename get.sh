#!/usr/bin/env bash
# One-liner bootstrap: first install AND update, same command, any directory:
#   curl -fsSL https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/get.sh | bash
# NO sudo in front, ever: install.sh calls sudo itself for the steps that need
# it, and under sudo every path it writes moves to /root (refused below).
# It installs the box: the lane that serves, the proxy and the cockpit, and
# every other lane on demand (its first Load from the cockpit builds it), and
# prints the cockpit URL when it is done.
# Pass install.sh options after `bash -s --`, e.g.:
#   curl -fsSL .../get.sh | bash -s -- --no-service
# Env overrides pass through (PORT=, HF_CACHE=, DIR= for a custom clone location).
# Everything is wrapped in main() so nothing runs on a partial download.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME

main() {
  # ── This does not run under sudo, and the refusal comes before the clone ──
  # Piped into `sudo bash`, every path below moves to /root: the clone lands in
  # /root/dgx-spark-qwen38 and install.sh then writes /root/.config/qwen38 with
  # a brand new API key, leaving a healthy engine that answers 401 to every
  # client reading ~/.config/qwen38/api-key. That happened on the reference box
  # on 2026-09-13. install.sh calls sudo itself, for the steps that need it.
  # install.sh refuses again on its own; this one exists so the refusal lands
  # before anything is written, and names the command that was actually typed.
  if [ "$(id -u)" = "0" ]; then
    if [ -n "${SUDO_USER:-}" ]; then
      printf '\n\033[1;31mERROR:\033[0m do not pipe this into "sudo bash" (your login is %s).\n\n' "$SUDO_USER" >&2
      printf '  run : curl -fsSL https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/get.sh | bash\n' >&2
      printf '        (no sudo in front: the installer calls sudo itself, for the steps that need it)\n\n' >&2
      printf 'Under sudo, HOME is /root: the clone and the API key land there, the units point\n' >&2
      printf 'at them, and every client reading ~/.config/qwen38/api-key gets 401 from an engine\n' >&2
      printf 'that installed, started and served perfectly well.\n' >&2
      exit 1
    fi
    if [ "${ALLOW_ROOT:-0}" != "1" ]; then
      printf '\n\033[1;31mERROR:\033[0m run this as the user who will use the box, not as root.\n' >&2
      printf '  everything it installs is addressed from $HOME, which as root is /root.\n' >&2
      printf '  if this box genuinely has no other user: ALLOW_ROOT=1 before the pipe.\n' >&2
      exit 1
    fi
  fi

  REPO_URL="https://github.com/hasso5703/dgx-spark-qwen38"
  DEFAULT_DIR="$HOME/dgx-spark-qwen38"
  # Is this origin a clone of THIS repo? The repository is renamed hasso5703/alzeo (planned before
  # 2026-11-02): every box installed before keeps the old name in its origin, which GitHub
  # redirects, and a clone made from the new address carries the new one. Both are this repo, for
  # good. The old name keeps the pattern it always had, now whatever the case, which only takes more,
  # so that no box installed before is refused.
  # The new name is matched as the last two parts of the path, HTTPS or SSH, with or without .git,
  # on any host as the old name always was (a mirror of this repository counts): hasso5703/alzeo-site,
  # or xhasso5703/alzeo, is another repository, which this script must not update or install from.
  # GitHub's names ignore case, and so does this: a clone of https://github.com/Hasso5703/Alzeo is
  # this repo too. Lowered by tr in the C locale, where every capital lowers as in English: bash 3.2
  # (macOS) takes ${1,,} for a bad substitution (code review, 2026-11-01).
  this_repo() {
    case "$(printf '%s' "$1" | LC_ALL=C tr '[:upper:]' '[:lower:]')" in
      *hasso5703/dgx-spark-qwen38*|*[/:]hasso5703/alzeo|*[/:]hasso5703/alzeo.git|*[/:]hasso5703/alzeo/|*[/:]hasso5703/alzeo.git/|*[/:]hasso5703/alzeo/.git|*[/:]hasso5703/alzeo/.git/) return 0 ;;
    esac
    return 1
  }
  command -v git >/dev/null || { echo "ERROR: git is required (stock on DGX OS)." >&2; exit 1; }

  # Case 1: running from inside an existing clone of this repo (any path) -> use it.
  DIR="${DIR:-}"
  if [ -z "$DIR" ] && TOP=$(git rev-parse --show-toplevel 2>/dev/null); then
    ORIGIN=$(git -C "$TOP" remote get-url origin 2>/dev/null || true)
    if this_repo "$ORIGIN"; then DIR="$TOP"; fi
  fi

  # Case 2: default location -> reuse the clone or create it.
  if [ -z "$DIR" ]; then
    DIR="$DEFAULT_DIR"
    if [ -d "$DIR/.git" ]; then
      :
    elif [ -e "$DIR" ]; then
      echo "ERROR: $DIR exists but is not a clone of this repo. Move it, or rerun with DIR=/path/to/clone" >&2
      exit 1
    else
      echo "── Cloning $REPO_URL into $DIR"
      git clone "$REPO_URL" "$DIR"
    fi
  fi

  # Whatever resolved DIR: it must be a clone of THIS repo before we touch it (this_repo above).
  DIR_ORIGIN=$(git -C "$DIR" remote get-url origin 2>/dev/null || true)
  if ! this_repo "$DIR_ORIGIN"; then
    # the origin as git keeps it, but for a user name and password or token in it, which a refusal
    # pasted into an issue would publish (code review, 2026-10-31). Everything up to the last @ goes,
    # after a remote helper's prefix if there is one (gcrypt::https://...), and the user of an
    # scp-like address: a password holding an @ or a / of its own goes too, and in the C locale one
    # holding a byte that is not UTF-8 (code review, 2026-11-01). An @ in a repository's own path,
    # which GitHub's names never hold, shows cut.
    SHOWN=$(printf '%s' "$DIR_ORIGIN" | LC_ALL=C sed -E -e 's#^(([A-Za-z0-9+.-]+::)?[A-Za-z][A-Za-z0-9+.-]*://).*@#\1#' \
                                                     -e 's#^[^/:@]+@([^/:]+:)#\1#')
    echo "ERROR: $DIR is not a clone of this repo (origin: ${SHOWN:-none}). Refusing to touch it." >&2; exit 1
  fi

  echo "── Repo: $DIR"
  git -C "$DIR" fetch -q origin main
  # The release tags too. A fetch of main alone brings none, and the cockpit's release check
  # and `git describe` read them: a box updated by this command kept naming the release it
  # was first cloned at (v1.19.0 on the reference box running v1.20.1, 2026-10-01). A tag of
  # the same name made on this box is kept, and the update goes on.
  git -C "$DIR" fetch -q --tags origin 2>/dev/null \
    || echo "── NOTE: the release tags did not all come (a tag of the same name exists here?); the update goes on"

  # Must be on main (a detached HEAD or a side branch would silently pin an old version).
  CUR=$(git -C "$DIR" symbolic-ref -q --short HEAD || echo DETACHED)
  if [ "$CUR" != "main" ] && [ "${FORCE_UPDATE:-0}" != "1" ]; then
    echo "ERROR: $DIR is on '$CUR', not 'main'; refusing to install from it." >&2
    echo "  rerun with FORCE_UPDATE=1 to switch to main (your branch/commit is kept)" >&2
    exit 1
  fi

  # Invariant: if this script completes, you ARE on the latest origin/main.
  # Only TRACKED modifications can make the checkout stale or altered. Untracked
  # files cannot, and `git reset --hard` below leaves them alone, so blocking on
  # them turned an unrelated notes file into a failed install and, under
  # FORCE_UPDATE, swept it into a stash the user never asked for.
  if [ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]; then
    if [ "${FORCE_UPDATE:-0}" = "1" ]; then
      echo "── FORCE_UPDATE=1: stashing your tracked changes (recoverable: git stash list)"
      git -C "$DIR" stash push -m "get.sh auto-stash before update"
    else
      echo "ERROR: $DIR has modified tracked files, refusing to install a stale or altered version." >&2
      git -C "$DIR" status --short --untracked-files=no | head -10 >&2
      echo "Fix with ONE of:" >&2
      echo "  keep your changes aside :  FORCE_UPDATE=1  then rerun this command (recover later: git -C $DIR stash pop)" >&2
      # not `git clean -fd` as well: it deletes the untracked files this script keeps
      # on purpose (found in review, 2026-09-24)
      echo "  discard your changes    :  git -C $DIR checkout -- .  then rerun" >&2
      exit 1
    fi
  fi
  # Off main under FORCE_UPDATE, the switch comes after the stash, and a detached HEAD's
  # commit is put on a branch first. Switched first, a change that conflicted with main
  # ended the run (rc 128), and a commit made on a detached HEAD was left on no branch
  # (found in review, 2026-09-24).
  if [ "$CUR" != "main" ]; then
    if [ "$CUR" = "DETACHED" ]; then
      BK="backup-$(git -C "$DIR" rev-parse --short HEAD)"
      git -C "$DIR" branch -f "$BK" >/dev/null
      echo "── FORCE_UPDATE=1: the detached commit is kept on branch '$BK'"
    fi
    echo "── FORCE_UPDATE=1: switching from '$CUR' to main (your branch is kept)"
    git -C "$DIR" checkout -q main 2>/dev/null || git -C "$DIR" checkout -qb main origin/main
  fi
  AHEAD=$(git -C "$DIR" rev-list --count origin/main..HEAD 2>/dev/null || echo 1)
  if [ "$AHEAD" != "0" ]; then
    if [ "${FORCE_UPDATE:-0}" = "1" ]; then
      BK="backup-$(git -C "$DIR" rev-parse --short HEAD)"
      git -C "$DIR" branch -f "$BK" >/dev/null
      echo "── FORCE_UPDATE=1: local commits kept on branch '$BK', resetting to origin/main"
    else
      echo "ERROR: $DIR has local commits not on origin/main, refusing to install a stale version." >&2
      echo "  keep your commits aside :  FORCE_UPDATE=1  then rerun (they stay on a backup branch)" >&2
      exit 1
    fi
  fi
  # Untracked files are kept, but say so: if one of them is a path that origin/main
  # now tracks, the reset below writes over it.
  UNTRACKED=$(git -C "$DIR" ls-files --others --exclude-standard)
  if [ -n "$UNTRACKED" ]; then
    echo "── Untracked files kept in place (not part of the update):"
    printf '%s\n' "$UNTRACKED" | head -5 | sed 's/^/     /'
  fi
  # Tree verified clean and no local commits (or backed up): hard sync is lossless
  # and, unlike ff-only, immune to shallow-clone ancestry gaps.
  git -C "$DIR" reset -q --hard origin/main
  echo "── At: $(git -C "$DIR" log -1 --format='%h %s')"

  cd "$DIR"
  # exec from a real file. stdin is still the pipe this script came through (read to
  # its end by now), so nothing in install.sh may read from it; sudo prompts on the tty
  # it opens itself.
  exec bash ./install.sh "$@"
}

main "$@"
