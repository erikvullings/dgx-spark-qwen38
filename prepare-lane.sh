#!/usr/bin/env bash
# The heavy half of a lane installed on demand, as the user, with no sudo: what the cockpit's
# first Load of the lane runs before its switch. The image and video lanes' runtime (a venv and
# SGLang Diffusion, the video lane's Turbo adapter) and Kolibri's image, each by its own
# installer's --prepare; their checkpoint comes with the Load's switch, as for any Load. The
# other text lane, beside the lane that serves: the image its launcher (flash) or unit (27B)
# runs, the checkpoint of the target it serves (the switch's own fetch, ./switch-model.sh
# <target> --fetch), and what that lane builds or keeps beside it: the flash lane's reduced
# draft vocabulary (lanes.sh), which its launcher takes at its next start, and the 27B's DFlash2
# draft, its config set for the window the unit serves. Each download is refused before its
# first byte when its room is not there.
#
#   ./prepare-lane.sh image|video|kolibri|flash|27b
#
# The lane is an argument of its own: the cockpit's actions never build one argument out of a
# parameter (tests/test_cockpit_http.py holds them to that).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage(){ echo "usage: ./prepare-lane.sh <image|video|kolibri|flash|27b>" >&2; exit 2; }
die(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
section(){ printf '\n\033[1;36m── %s\033[0m\n' "$*"; }
[ "$#" -eq 1 ] || usage

# What a launcher or unit says. head -1 each time: -m1 stops at the first line, and -o prints
# every match on it (the flash lane's draft vocabulary line names its file twice).
arg_of(){  # $1 file, $2 flag: the value the file gives that flag
  { grep -m1 -oE -- "$2 [^ ]+" "$1" || true; } | head -1 | cut -d' ' -f2
}
mount_of(){  # $1 file, $2 the container path: the host folder mounted there
  { grep -m1 -oE -- "-v [^ :]+:$2( |$)" "$1" || true; } | head -1 | sed -e 's/^-v //' -e "s|:$2 *\$||"
}
room_or_die(){  # $1 GB, $2 a path on the disk it goes to, $3 what for
  local free; free="$(free_gb_at "$2")"
  [ "${free:-0}" -ge "$1" ] || die "need ~$1 GB free under $2 for $3; found ${free:-unknown} GB. Free some space, then Load the lane again: what came already is kept."
}
pull_image(){  # $1 image, $2 the lane (its local tag), $3 the room its pull needs in GB
  section "The $2 lane's image (${1:0:40})"
  if docker image inspect "$1" >/dev/null 2>&1; then
    echo "already here"
  else
    room_or_die "$3" "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)" "its image"
    docker pull "$1" || die "docker pull of $1 failed (no internet, or a Docker Hub rate limit: retry in a few minutes)"
  fi
  pin_tag "$2" "$1"
}
pins_or_die(){  # the pins install.sh declares, the one place they are, each checked defined
  local v
  eval "$(grep -E '^(FLASH(_NVDA|_UNC)?|STOCK|UNC|FP8|UNCFP8)_(REPO|REV)=' "$HERE/install.sh" || true)"
  for v in FLASH_REPO FLASH_REV FLASH_NVDA_REPO FLASH_NVDA_REV FLASH_UNC_REPO FLASH_UNC_REV \
           STOCK_REPO STOCK_REV UNC_REPO UNC_REV FP8_REPO FP8_REV UNCFP8_REPO UNCFP8_REV; do
    eval "[ -n \"\${$v:-}\" ]" || die "install.sh no longer defines $v (repo layout changed?)"
  done
}

prepare_flash(){
  local launch="$CONFIG_DIR/launch-flash.sh" img model hf ple map target rev
  [ -f "$launch" ] || die "the flash lane is not installed on this box (no $launch): ./install.sh installs it"
  img="$(engine_image "$launch")"; model="$(arg_of "$launch" --model-path)"; hf="$(mount_of "$launch" /root/.cache/huggingface)"
  ple="$(mount_of "$launch" /ple)"
  if [ -z "$img" ] || [ -z "$model" ] || [ -z "$hf" ]; then
    die "$launch names no image, model or cache where this expects it (hand-edited?): ./install.sh writes it again"
  fi
  case "$model" in
    "$FLASH_REPO") target=flash; rev="$FLASH_REV" ;;
    "$FLASH_NVDA_REPO") target=flash-nvda; rev="$FLASH_NVDA_REV" ;;
    "$FLASH_UNC_REPO") target=flash-uncensored; rev="$FLASH_UNC_REV" ;;
    *) die "the flash launcher serves $model, which no flash target names: ./switch-model.sh flash points it at one" ;;
  esac
  pull_image "$img" flash 35
  bash "$HERE/switch-model.sh" "$target" --fetch
  # The PLE table its launcher writes at every boot, 47.7 GiB in PLE_DIR, needs its room there,
  # counted after the checkpoint when the two share a disk (install.sh asks for both at once).
  if [ -n "$ple" ] && ! ls "$ple"/ple_table_*.bin >/dev/null 2>&1; then
    room_or_die 50 "$ple" "the flash lane's PLE table (47.7 GiB, written at every boot)"
  fi
  map="$({ grep -m1 -oE 'token-map-[0-9]+\.pt' "$launch" || true; } | head -1)"
  if [ -n "$map" ]; then
    section "The flash lane's reduced draft vocabulary"
    map="${map#token-map-}"
    flash_token_map "$img" "$hf" "$model" "$rev" "${map%.pt}" "its checkpoint did not come: re-run ./prepare-lane.sh flash" \
      || die "building the reduced draft vocabulary failed (the lines above say why); the lane serves without it"
  fi
  printf '\n\033[1;32m── Done: the flash lane is prepared; its Load switches to it and starts it\033[0m\n'
}

prepare_27b(){
  local unit=/etc/systemd/system/qwen38-sglang.service img model draft drev hf target tok=()
  [ -f "$unit" ] || die "the 27B lane is not installed on this box (no $unit): ./install.sh installs it"
  img="$(engine_image "$unit")"; model="$(arg_of "$unit" --model-path)"; hf="$(mount_of "$unit" /root/.cache/huggingface)"
  draft="$(arg_of "$unit" --speculative-draft-model-path)"; drev="$(arg_of "$unit" --speculative-draft-model-revision)"
  if [ -z "$img" ] || [ -z "$model" ] || [ -z "$hf" ]; then
    die "$unit names no image, model or cache where this expects it (hand-edited?): ./install.sh writes it again"
  fi
  case "$model" in
    "$STOCK_REPO") target=stock ;;
    "$UNC_REPO") target=uncensored ;;
    "$FP8_REPO") target=fp8 ;;
    "$UNCFP8_REPO") target=uncensored-fp8 ;;
    *) die "the 27B unit serves $model, which no 27B target names: ./switch-model.sh stock points it at one" ;;
  esac
  pull_image "$img" 27b 40
  if [ -n "$draft" ]; then
    section "The 27B lane's DFlash2 draft ($draft @ ${drev:-main})"
    room_or_die 5 "$hf" "its DFlash2 draft"
    # in the lane's own image, as install.sh fetches it; a token goes by name, never on a command line
    if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; tok=(-e HF_TOKEN); fi
    docker run --rm --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
      -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 ${tok[@]+"${tok[@]}"} \
      -v "$HERE/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" -v "$hf":/hf \
      "$img" /opt/qwen38/ckpt-fetch.py "$draft" "${drev:-main}" --refs-main fill \
      || die "the DFlash2 draft did not come (no internet, or HuggingFace throttling: set HF_TOKEN, or Load again: downloads resume)"
    # Its config carries the YaRN patch when the unit serves the 1M window, or the draft crashes
    # at load; the switch patches the target's (install.sh does both, at once).
    if grep -q -- '--context-length 1010000' "$unit"; then
      python3 "$HERE/patch-yarn.py" "$hf" "$draft" "${drev:-main}" || die "YaRN patch failed on the DFlash2 draft"
    else
      python3 "$HERE/patch-yarn.py" --restore "$hf" "$draft" "${drev:-main}" || die "YaRN restore failed on the DFlash2 draft"
    fi
  fi
  bash "$HERE/switch-model.sh" "$target" --fetch
  printf '\n\033[1;32m── Done: the 27B lane is prepared; its Load switches to it and starts it\033[0m\n'
}

case "$1" in
  image|video|kolibri) exec bash "$HERE/install-$1.sh" --prepare ;;
  flash|27b)
    CONFIG_DIR="$HOME/.config/qwen38"; REPO_DIR="$HERE"
    # shellcheck source=lanes.sh
    . "$HERE/lanes.sh"
    pins_or_die
    if [ "$1" = flash ]; then prepare_flash; else prepare_27b; fi ;;
  *) usage ;;
esac
