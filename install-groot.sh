#!/usr/bin/env bash
# Install the experimental Groot-65B-A6B mixed NVFP4/FP8 lane on a 128 GB Spark.
# Usage: ./install-groot.sh [--no-smoke|--units-only|--prepare|--uninstall]
# --stage FILE renders a unit without sudo, for switch-model.sh.
# The shared b12x image supplies GB10-compatible vLLM; this lane uses the standard
# compressed-tensors loader, no speculative decoder and no context extension.
# All memory/context choices are conservative starting settings, not measurements.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-groot.service
# where systemd reads the units, the one variable lanes.sh and vllm-lane.sh read the other units by
INSTALLED="${LANES_UNIT_DIR:-/etc/systemd/system}/$UNIT"
CONTAINER=qwen38-groot
die(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
step(){ printf '\n\033[1;36m── %s\033[0m\n' "$*"; }

# Under sudo every path below moves to /root: the cache, the config and the key the unit
# reads. install.sh refuses the same way, for the same reason.
if [ "$(id -u)" = "0" ] && [ "${ALLOW_ROOT:-0}" != "1" ]; then
  die "run this as the user who will use the box, not as root (everything it installs is addressed from \$HOME)."
fi

SMOKE=1; ACTION=install; MODE=full
for a in "$@"; do case "$a" in
  --no-smoke) SMOKE=0 ;;
  --uninstall) ACTION=uninstall ;;
  # The two halves of a lane installed on demand, as for the other lanes: --units-only puts in
  # what needs root (the unit, its start scripts) and nothing heavy, and --prepare pulls the
  # lane's image as the user, with no sudo, when the cockpit first loads the lane. The checkpoint
  # comes with that Load's switch, as for any Load.
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  --stage) [ "$MODE" = full ] || die "--stage is a mode of its own"; MODE=stage ;;
  -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
  *) if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then STAGE_OUT="$a"; else die "unknown option: $a"; fi ;;
esac; done
if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then die "--stage takes the file to write the unit to"; fi

CONFIG_DIR="$HOME/.config/qwen38"
# the text lane the box comes back to, one record shared with switch-model.sh (lanes.sh)
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"
# A 64 GB DGX Spark cannot hold this lane (boxmem.py): refused there before anything is fetched,
# and --uninstall still takes away what an older release put there (found in review, 2026-10-23).
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits groot || die "The Groot 65B A6B lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# what the vLLM lanes' installers share: the engine's cache and port, the uninstall, the service,
# the preflight and the room, the fetch, the proof
# shellcheck source=vllm-lane.sh
. "$HERE/vllm-lane.sh"
VLANE=groot; VLANE_SHORT="Groot 65B A6B"

# Quantized checkpoint revision resolved on 2026-10-31. Compressed-tensors mixed NVFP4/FP8, 98.6 GB on its model card.
GROOT_REPO="GrootTeam/Groot-65B-A6B-261023-NVFP4"
GROOT_REV="68a352a55f51590ce0d7741a5d001d0555b34d09"
GROOT_GMU="0.85"
GROOT_GB="100"
GROOT_SEQS="1"
MODEL="$GROOT_REPO"; MODEL_REV="$GROOT_REV"
GMU="$GROOT_GMU"; WEIGHTS_GB="$GROOT_GB"; SEQS="$GROOT_SEQS"
# The image is the Flash FP4-Local lane's, by its pin in install-flash-local.sh, the one place it is
# declared, pulled as published, by digest: nothing is built on the box. Its pull at its peak, as
# that installer counts it (its IMAGE_GB, read there).
eval "$(grep -E '^FLASH_LOCAL_IMAGE=' "$HERE/install-flash-local.sh" || true)"
[ -n "${FLASH_LOCAL_IMAGE:-}" ] || die "install-flash-local.sh no longer defines FLASH_LOCAL_IMAGE (repo layout changed?)"
LOCAL_IMAGE="$FLASH_LOCAL_IMAGE"
IMAGE_GB="$({ grep -m1 -oE '^IMAGE_GB=[0-9]+' "$HERE/install-flash-local.sh" || true; } | cut -d= -f2)"
case "$IMAGE_GB" in ''|*[!0-9]*) die "install-flash-local.sh no longer defines IMAGE_GB (repo layout changed?)" ;; esac

# The text lanes' cache and port: Nanbeige is one of them for the proxy (vllm-lane.sh).
vlane_engine_env "${GROOT_PORT:-}"

if [ "$ACTION" = uninstall ]; then
  step "Removing the Groot 65B A6B lane"
  vlane_uninstall_unit
  vlane_uninstall_clients
  # The image is pulled, not built: no build of it to prune (qwen38-groot names none), and
  # its local tag goes alone (lanes.sh), the Flash FP4-Local lane's own staying.
  vlane_uninstall_runtime qwen38-groot "$FLASH_LOCAL_IMAGE" "$GROOT_REPO"
  pin_untag groot
  rm -f "$CONFIG_DIR/chat-template-groot.jinja"
  exit 0
fi

# The unit, rendered for this box: written to $1.
render_unit(){
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$LOCAL_IMAGE|g" -e "s|__MODEL__|$MODEL|g" -e "s|__REV__|$MODEL_REV|g" \
      -e "s|__GMU__|$GMU|g" -e "s|__SEQS__|$SEQS|g" \
      -e "s|__HF_CACHE__|$HF_CACHE|g" -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$1"
}

# The official checkpoint chat template, without model-family patches.
TEMPLATE="$CONFIG_DIR/chat-template-groot.jinja"
lane_template(){
  python3 - "$HF_CACHE" "$MODEL" "$MODEL_REV" "$TEMPLATE" <<'PYT'
import json, pathlib, sys
cache, model, rev, out = sys.argv[1:]
snap = pathlib.Path(cache) / "hub" / ("models--" + model.replace("/", "--")) / "snapshots" / rev
src = snap / "chat_template.jinja"
text = src.read_text() if src.is_file() else json.loads((snap / "tokenizer_config.json").read_text())["chat_template"]
if not isinstance(text, str) or not text.strip():
    sys.exit("Pinned checkpoint has no usable chat template")
p = pathlib.Path(out)
if not p.exists() or p.read_text() != text:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(p)
PYT
}

# Its own vLLM compile cache.
CACHES=("$CONFIG_DIR/groot-cache/vllm")

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a lane
# installed on demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  vlane_service_files "${CACHES[@]}"
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  render_unit "$RENDER"
  # with what the engine reads at its start besides the unit: a lane started before they changed
  # serves the old ones (vllm-lane.sh)
  vlane_put_unit "$RENDER" "$TEMPLATE"
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Groot 65B A6B in the Lanes view"
  echo "  a terminal : ./switch-model.sh groot, then the commands it prints"
}

# --stage: the unit for the checkpoint in the cache, for switch-model.sh, with no sudo. This lane's
# own files only, as the other lanes' switches touch nothing but their unit: the start scripts every
# lane runs (vlane_service_files) are ./install.sh's to put in (the Flash FP4-Local lane's rule). Its
# caches made as the user, so that docker does not make them as root, and its filter only when it
# is missing.
if [ "$MODE" = stage ]; then
  lane_template >&2
  mkdir -p "${CACHES[@]}"
  render_unit "$STAGE_OUT"
  grep -q '__[A-Z][A-Z0-9_]*__' "$STAGE_OUT" && die "the unit template still holds an unsubstituted placeholder"
  exit 0
fi

step "1/5 Preflight (Groot 65B A6B: $MODEL @ ${MODEL_REV:0:12})"
vlane_preflight engine-preflight.sh engine-secrets.sh vllm_engine_auth.py ckpt-fetch.py \
  install-flash-local.sh
vlane_room "$FLASH_LOCAL_IMAGE" "$MODEL"
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the checkpoint come with the lane's first Load"
  if ckpt_cached "$HF_CACHE/hub/models--${MODEL//\//--}" "$MODEL_REV"; then lane_template; fi
  install_service
  step "Done: the Groot 65B A6B lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Groot 65B A6B in the Lanes view pulls its image (the Flash FP4-Local lane's,"
  echo "               about 18 GB, 10 to download, once for the two), fetches its checkpoint (about $WEIGHTS_GB GB), then starts it"
  echo "  a terminal : ./install-groot.sh --prepare, then ./switch-model.sh groot and the commands it prints"
  exit 0
fi

step "2/5 Image (${FLASH_LOCAL_IMAGE:0:40}..., the Flash FP4-Local lane's)"
if docker image inspect "$FLASH_LOCAL_IMAGE" >/dev/null 2>&1; then
  echo "already here"
else
  docker pull "$FLASH_LOCAL_IMAGE" \
    || die "docker pull of $FLASH_LOCAL_IMAGE failed. Causes: no internet, a Docker Hub rate limit (retry in a few minutes, or docker login), or the digest removed upstream."
fi
# its local tag, so that a docker image prune leaves it alone while this lane is installed, and the
# tags of the images it was pinned to before go (lanes.sh); the Flash FP4-Local lane keeps its own
pin_tag groot "$FLASH_LOCAL_IMAGE"
unpin_stale groot "$FLASH_LOCAL_IMAGE" || true

if [ "$MODE" = prepare ]; then
  step "Done: the Groot 65B A6B lane's image is ready; its Load fetches the checkpoint and starts it"
  exit 0
fi

step "3/5 Checkpoint ($MODEL @ ${MODEL_REV:0:12}, resumable) and its chat template"
vlane_fetch "$MODEL" "$MODEL_REV"
lane_template

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves (a reply and a tool call, then the box goes back to the lane it was serving)"
VLANE_SERVED=groot-65b-a6b; VLANE_KWARGS='{"enable_thinking":false,"preserve_thinking":true}'
VLANE_PARSERS="the qwen3_coder parser is"; VLANE_BOOT="not yet measured on GB10"; VLANE_WAIT_MIN=40
vlane_prove

step "Done: the Groot 65B A6B lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  switch to it : Load on Groot 65B A6B in the cockpit, or ./switch-model.sh groot"
echo "  clients      : the proxy on :30001 with the box's key, model groot-65b-a6b (OpenAI and Anthropic APIs)"
