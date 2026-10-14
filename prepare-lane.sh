#!/usr/bin/env bash
# The heavy half of a lane installed on demand, as the user, with no sudo: what the cockpit's
# first Load of the lane runs before its switch. The image and video lanes' runtime (a venv and
# SGLang Diffusion, the video lane's Turbo adapter) and Kolibri's image, each by its own
# installer's --prepare; their checkpoint comes with the Load's switch, as for any Load. The
# flash lane beside the lane that serves: the image its launcher names, the checkpoint of the
# target it serves (the switch's own fetch, ./switch-model.sh <target> --fetch) and the reduced
# draft vocabulary built from it (lanes.sh), which its launcher takes at its next start.
#
#   ./prepare-lane.sh image|video|kolibri|flash
#
# The lane is an argument of its own: the cockpit's actions never build one argument out of a
# parameter (tests/test_cockpit_http.py holds them to that).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage(){ echo "usage: ./prepare-lane.sh <image|video|kolibri|flash>" >&2; exit 2; }
die(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 1 ] || usage

prepare_flash(){
  CONFIG_DIR="$HOME/.config/qwen38"; REPO_DIR="$HERE"
  # shellcheck source=lanes.sh
  . "$HERE/lanes.sh"
  local launch="$CONFIG_DIR/launch-flash.sh" img model hf map target rev
  [ -f "$launch" ] || die "the flash lane is not installed on this box (no $launch): ./install.sh installs it"
  # head -1 each time: -m1 stops at the first line, and -o prints every match on it (the draft
  # vocabulary's line names its file twice)
  img="$({ grep -m1 -oE 'lmsysorg/sglang@sha256:[0-9a-f]{64}|qwen38-flash:[A-Za-z0-9._-]+' "$launch" || true; } | head -1)"
  model="$({ grep -m1 -oE -- '--model-path [^ ]+' "$launch" || true; } | head -1 | cut -d' ' -f2)"
  hf="$({ grep -m1 -oE -- '-v [^ :]+:/root/\.cache/huggingface' "$launch" || true; } | head -1 | sed -e 's/^-v //' -e 's|:/root/\.cache/huggingface$||')"
  if [ -z "$img" ] || [ -z "$model" ] || [ -z "$hf" ]; then
    die "$launch names no image, model or cache where this expects it (hand-edited?): ./install.sh writes it again"
  fi
  # the target it serves, by the pins install.sh declares, the one place they are
  eval "$(grep -E '^FLASH(_NVDA|_UNC)?_(REPO|REV)=' "$HERE/install.sh" || true)"
  for v in FLASH_REPO FLASH_REV FLASH_NVDA_REPO FLASH_NVDA_REV FLASH_UNC_REPO FLASH_UNC_REV; do
    eval "[ -n \"\${$v:-}\" ]" || die "install.sh no longer defines $v (repo layout changed?)"
  done
  case "$model" in
    "$FLASH_REPO") target=flash; rev="$FLASH_REV" ;;
    "$FLASH_NVDA_REPO") target=flash-nvda; rev="$FLASH_NVDA_REV" ;;
    "$FLASH_UNC_REPO") target=flash-uncensored; rev="$FLASH_UNC_REV" ;;
    *) die "the flash launcher serves $model, which no flash target names: ./switch-model.sh flash points it at one" ;;
  esac
  printf '\n\033[1;36m── The flash lane'"'"'s image (%s)\033[0m\n' "${img:0:40}"
  if docker image inspect "$img" >/dev/null 2>&1; then
    echo "already here"
  else
    docker pull "$img" || die "docker pull of $img failed (no internet, or a Docker Hub rate limit: retry in a few minutes)"
  fi
  pin_tag flash "$img"
  bash "$HERE/switch-model.sh" "$target" --fetch
  map="$({ grep -m1 -oE 'token-map-[0-9]+\.pt' "$launch" || true; } | head -1)"
  if [ -n "$map" ]; then
    printf '\n\033[1;36m── The flash lane'"'"'s reduced draft vocabulary\033[0m\n'
    map="${map#token-map-}"
    flash_token_map "$img" "$hf" "$model" "$rev" "${map%.pt}" "its checkpoint did not come: re-run ./prepare-lane.sh flash" \
      || die "building the reduced draft vocabulary failed (the lines above say why); the lane serves without it"
  fi
  printf '\n\033[1;32m── Done: the flash lane is prepared; its Load switches to it and starts it\033[0m\n'
}

case "$1" in
  image|video|kolibri) exec bash "$HERE/install-$1.sh" --prepare ;;
  flash) prepare_flash ;;
  *) usage ;;
esac
