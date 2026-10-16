#!/usr/bin/env bash
# Install the Nemotron lane: NVIDIA's Nemotron 3.5 Bolt, a mixture of experts of 74B parameters
# (9B active per token) compressed to 24 GB (its experts in 2 bits), with reasoning, tool calling
# and image and video input, served by vLLM on the text lanes' port behind the proxy and its key.
# Idempotent, and safe to run alone.
#
#   ./install-nemotron.sh                 first install, or update in place
#   ./install-nemotron.sh --no-smoke      skip the proof at the end
#   ./install-nemotron.sh --uninstall     remove the unit and the local image (weights kept)
#   ./install-nemotron.sh --units-only    the unit alone: no image, no weights (install.sh, on demand)
#   ./install-nemotron.sh --prepare       its image, built with no sudo: the cockpit's first Load
#                                         runs it (./prepare-lane.sh nemotron)
#
# install.sh puts this lane's unit in on every plain install, like the other lanes, and its first
# Load from the cockpit builds it; ./install.sh --no-nemotron leaves it out for good (a marker file
# remembers it). It takes 26 GB of weights (the model and its drafter) and a 25 GB image, 35 at its
# pull's peak; when the disk does not have the room, this refuses before downloading anything.
#
# WHY THIS BUILD. NVIDIA's model card serves Bolt with one vLLM nightly, 77154ca6, plus the b12x
# kernel library (its 2-bit IQ2_XXS experts, NVFP4 Mamba layers and Q8_0 layers run on b12x on the
# GB10, SM121). That image is published for arm64; this builds on it, pinned by the digest of its
# arm64 manifest, plus b12x's wheel checked against the sha256 PyPI publishes, installed with
# --no-deps (the image already carries every dependency b12x names, checked with pip check). The
# card's other step, `vllm download-kernels`, installs FlashInfer's prebuilt kernels: this image
# already holds exactly those (flashinfer-cubin and flashinfer-jit-cache 0.7.0.post1, sm121a among
# them), so it is not run, and the lane starts with no internet.
#
# ONE ENGINE AT A TIME. The unit says so with Conflicts=: starting it stops whichever lane
# serves, and starting another lane stops it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-nemotron.service
INSTALLED="/etc/systemd/system/$UNIT"
CONTAINER=qwen38-nemotron
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
  # The two halves of a lane installed on demand, as for the image, video and Kolibri lanes:
  # --units-only puts in what needs root (the unit, its start scripts) and nothing heavy, and
  # --prepare builds the lane's image as the user, with no sudo, when the cockpit first loads
  # the lane. The checkpoints come with that Load's switch, as for any Load.
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
  *) die "unknown option: $a" ;;
esac; done

CONFIG_DIR="$HOME/.config/qwen38"
# the text lane the box comes back to, one record shared with switch-model.sh (lanes.sh)
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"

# The pins, at column 0 like install.sh's: the vLLM nightly NVIDIA's card names (its arm64
# manifest), the b12x wheel, the model and its drafter at the revisions measured on the reference
# box on 2026-10-15. NVIDIA rewrote both repos' history when it made them public ("initial commit",
# 15:42 UTC): the revisions measured, from before, are gone, and these hold every one of the same
# files bit for bit (blob hashes compared), the model card and one note excepted.
NEMOTRON_BASE_IMAGE="vllm/vllm-openai@sha256:1ee85ca80e0fc110017d79ea2081a0cbef9a208a05d7f22724801bcf6a956620"
NEMOTRON_VLLM_LABEL="nightly-77154ca6"
NEMOTRON_B12X_VERSION="1.5.1"
NEMOTRON_B12X_URL="https://files.pythonhosted.org/packages/0c/7f/31fbc8311188835a509d97c38f6400071ba5b01a5ca54cb33d7b22dec3fa/b12x-1.5.1-py3-none-any.whl"
NEMOTRON_B12X_SHA256="8731e23571d8d45db26d78e7699ec0b23e7ba20b52487adf8c1063c8d8b53a3d"
NEMOTRON_BOLT_REPO="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
NEMOTRON_BOLT_REV="c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"
NEMOTRON_BOLT_DRAFT_REPO="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
NEMOTRON_BOLT_DRAFT_REV="4c6a5613b6d887194a3a5493e3a18a5b3930bb86"
# The share of the GPU memory and the download in GB (why: where they are used below).
# switch-model.sh reads them here, with the repos and revisions.
NEMOTRON_BOLT_GMU="0.70"
NEMOTRON_BOLT_GB="26"
# The local image: the base and the wheel, nothing else. Its tag names both, so a change of
# either pin is a new tag, and a box never serves an image built from older pins. The wheel is
# named by its version, which PyPI never lets a file change under.
B12X_WHEEL="b12x-${NEMOTRON_B12X_VERSION}-py3-none-any.whl"
[ "${NEMOTRON_B12X_URL##*/}" = "$B12X_WHEEL" ] \
  || die "NEMOTRON_B12X_URL names ${NEMOTRON_B12X_URL##*/}, not the $B12X_WHEEL of NEMOTRON_B12X_VERSION (repo bug: please open an issue)"
LOCAL_IMAGE="qwen38-nemotron:vllm${NEMOTRON_VLLM_LABEL#nightly-}-b12x${NEMOTRON_B12X_VERSION}-${NEMOTRON_BASE_IMAGE##*:}"
LOCAL_IMAGE="${LOCAL_IMAGE:0:52}"

# One checkpoint today, Bolt with its LiLiCorr drafter. Its share of the GPU memory, 0.70, holds a
# KV cache of 4,712,014 tokens (measured, 2026-10-15): 18 requests at the full 262,144 window,
# for the 16 the unit lets in at once, with 26 GiB of MemAvailable left to the box at rest (25.3 at
# its least, a 252,000-token prompt's prefill). The drafter
# costs that cache dearly (2.0 million tokens at 0.45 with it, 5.2 without) and pays it back at
# every load measured (72.7 against 30.3 tok/s for one stream, 213 against 178 for sixteen).
MODEL="$NEMOTRON_BOLT_REPO"; MODEL_REV="$NEMOTRON_BOLT_REV"
DRAFT="$NEMOTRON_BOLT_DRAFT_REPO"; DRAFT_REV="$NEMOTRON_BOLT_DRAFT_REV"
GMU="$NEMOTRON_BOLT_GMU"; WEIGHTS_GB="$NEMOTRON_BOLT_GB"
# The base image's pull at its peak: 24.46 GB unpacked with the wheel's layer (27 MB) and 10.30 GB
# to download (the registry's arm64 manifest), which the containerd image store (Docker 29's
# default on a new install) keeps beside it, 32.4 GiB; ceil(x 1.02) + 1 as df -BG rounds up
# (install.sh counts the text lanes' images so). It asked for 34, from rounded sizes (found in
# review, 2026-10-15).
IMAGE_GB=35

# The text lanes' cache and port: Nemotron is one of them for the proxy, so it answers where
# the proxy sends (its UPSTREAM), on the bind the other lanes use, from the same cache.
TEXT_UNITS="${TEXT_UNITS:-/etc/systemd/system/qwen38-sglang.service $CONFIG_DIR/launch-flash.sh}"
text_cache(){
  # shellcheck disable=SC2086  # a list of paths
  { grep -hoE -- '-v [^ :]+:/root/\.cache/huggingface' $INSTALLED $TEXT_UNITS 2>/dev/null || true; } \
    | head -1 | sed -e 's/^-v //' -e 's|:/root/\.cache/huggingface$||'
}
HF_CACHE="${HF_CACHE:-$(text_cache)}"; HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
text_flag(){
  # shellcheck disable=SC2086  # a list of paths
  { grep -hoE -- "$1 [^ \\\\']+" $INSTALLED $TEXT_UNITS 2>/dev/null || true; } | head -1 | awk '{print $2}'
}
UPSTREAM_PORT="$({ grep -m1 -oE 'UPSTREAM=http://[^:]+:[0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | awk -F: '{print $NF}')"
PORT="${NEMOTRON_PORT:-${UPSTREAM_PORT:-$(text_flag --port)}}"; PORT="${PORT:-30000}"
ENGINE_BIND="${ENGINE_BIND:-$(text_flag --host)}"; ENGINE_BIND="${ENGINE_BIND:-127.0.0.1}"
case "$PORT" in ''|*[!0-9]*) die "the engine port must be a number, got: $PORT" ;; esac

if [ "$ACTION" = uninstall ]; then
  step "Removing the Nemotron lane"
  # what a box switched before v1.30 has in its links becomes the record first (lanes.sh)
  text_lane_migrate
  if [ -f "$INSTALLED" ]; then
    WAS_BOOT=0; systemctl is-enabled --quiet "$UNIT" 2>/dev/null && WAS_BOOT=1
    sudo systemctl disable --now "$UNIT" 2>/dev/null || true
    sudo rm -f "$INSTALLED"; sudo systemctl daemon-reload
    echo "unit removed"
    # Nemotron was the lane at boot: the text lane it replaced was disabled by that switch,
    # and removing this one would leave no engine at the next boot.
    if [ "$WAS_BOOT" -eq 1 ]; then
      BACK="$(text_lane_back nemotron)"
      if [ -n "$BACK" ]; then
        sudo systemctl enable "$BACK"
        echo "$BACK is enabled at boot again, as it was before Nemotron; start it now with: sudo systemctl start $BACK   (or the cockpit)"
      else
        echo "NOTE: no engine is enabled at boot any more. Start a lane from the cockpit, or:"
        echo "      sudo systemctl enable --now qwen38-sglang.service"
      fi
    fi
  fi
  # opencode and pi: the nemotron provider goes, and a default that named it follows the text
  # lane this box comes back to (lanes.sh; oc-point-default.py with no target: no name changes).
  case "$(text_lane_back nemotron)" in
    qwen38-flash.service) OC_LANE=flash ;;
    *) OC_LANE=27b ;;
  esac
  OC_SUM_BEFORE="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
  for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
    [ -f "$OC_JSON" ] || continue
    WAS_DEFAULT=0
    grep -qsE '"model"[[:space:]]*:[[:space:]]*"nemotron/' "$OC_JSON" && WAS_DEFAULT=1
    python3 "$HERE/oc-merge-limits.py" "$OC_JSON" --remove-provider nemotron \
      || echo "NOTE: could not take the nemotron provider out of $OC_JSON; remove it by hand"
    if [ "$WAS_DEFAULT" -eq 1 ]; then
      python3 "$HERE/oc-point-default.py" "$OC_JSON" "$OC_LANE" "" "" \
        || echo "NOTE: could not point $OC_JSON at the $OC_LANE lane; set its \"model\" field yourself"
      # The compaction with it, from the window of the model put back.
      OC_DEF_CTX="$(python3 - "$HERE/oc-merge-limits.py" "$OC_JSON" <<'PYCTX' || true
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ocm", sys.argv[1])
ocm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocm)
j = ocm.load(open(sys.argv[2], encoding="utf-8").read())
prov, _, mid = str(j.get("model") or "").partition("/")
entry = ((j.get("provider") or {}).get(prov) or {}).get("models", {}).get(mid) or {}
ctx = (entry.get("limit") or {}).get("context") if isinstance(entry, dict) else None
print(ctx if isinstance(ctx, int) and ctx > 0 else "")
PYCTX
)"
      OC_DEF_KEEP=""
      [ -n "$OC_DEF_CTX" ] && OC_DEF_KEEP="$("$HERE/oc-limits.sh" --preserve "$OC_DEF_CTX" 2>/dev/null || true)"
      if [ -n "$OC_DEF_KEEP" ]; then
        python3 "$HERE/oc-merge-limits.py" "$OC_JSON" --compaction "$OC_DEF_KEEP" \
          || echo "NOTE: could not size the compaction of $OC_JSON for the $OC_LANE lane ($OC_DEF_KEEP tokens); set it yourself"
      fi
    fi
  done
  if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
    python3 "$HERE/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
      || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
  fi
  if [ "$OC_SUM_BEFORE" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
     && systemctl is-active --quiet opencode-web.service 2>/dev/null; then
    sudo systemctl restart opencode-web.service && echo "opencode-web.service restarted so it reads the change"
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  for img in $(docker images --format '{{.Repository}}:{{.Tag}}' 'qwen38-nemotron' 2>/dev/null); do
    docker rmi "$img" >/dev/null 2>&1 && echo "image removed: $img"
  done
  # vLLM writes its caches as root inside the container: what the user cannot remove, sudo does
  rm -rf "$CONFIG_DIR/nemotron-cache" 2>/dev/null || sudo rm -rf --one-file-system "$CONFIG_DIR/nemotron-cache"
  # The key's middleware is the Kolibri lane's too: it stays while that lane is installed.
  if [ -f /etc/systemd/system/qwen38-kolibri.service ]; then
    echo "vllm-auth kept: the Kolibri lane uses it too"
  else
    rm -rf "$CONFIG_DIR/vllm-auth"
  fi
  echo "the base image $NEMOTRON_BASE_IMAGE stays (docker rmi it yourself if nothing else uses it)"
  for R in "$NEMOTRON_BOLT_REPO" "$NEMOTRON_BOLT_DRAFT_REPO"; do
    D="$HF_CACHE/hub/models--${R//\//--}"
    [ -d "$D" ] && echo "checkpoint kept, delete it yourself for the space: rm -rf $D"
  done
  exit 0
fi

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a
# lane installed on demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  # vLLM's compile cache, FlashInfer's kernels built at their first use and b12x's: made here as
  # this user, so that docker does not make them as root when it mounts them
  mkdir -p "$CONFIG_DIR" "$CONFIG_DIR/nemotron-cache/vllm" "$CONFIG_DIR/nemotron-cache/flashinfer" "$CONFIG_DIR/nemotron-cache/b12x"
  # The unit runs both before every start: run on its own, this installer puts them in place.
  cmp -s "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh" \
    || install -m 755 "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh"
  cmp -s "$HERE/engine-secrets.sh" "$CONFIG_DIR/engine-secrets.sh" \
    || install -m 755 "$HERE/engine-secrets.sh" "$CONFIG_DIR/engine-secrets.sh"
  # The middleware that asks the key on every route of the engine (see vllm_engine_auth.py),
  # in a directory of its own: the unit mounts that one, read-only, on the module path.
  mkdir -p "$CONFIG_DIR/vllm-auth"
  cmp -s "$HERE/vllm_engine_auth.py" "$CONFIG_DIR/vllm-auth/vllm_engine_auth.py" \
    || install -m 644 "$HERE/vllm_engine_auth.py" "$CONFIG_DIR/vllm-auth/vllm_engine_auth.py"
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn)|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$LOCAL_IMAGE|g" -e "s|__MODEL_DIR__|${MODEL//\//--}|g" -e "s|__MODEL__|$MODEL|g" \
      -e "s|__REV__|$MODEL_REV|g" -e "s|__DRAFT__|$DRAFT|g" -e "s|__DREV__|$DRAFT_REV|g" \
      -e "s|__GMU__|$GMU|g" -e "s|__HF_CACHE__|$HF_CACHE|g" \
      -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$RENDER"
  grep -q '__[A-Z][A-Z0-9_]*__' "$RENDER" && die "the unit template still holds an unsubstituted placeholder: $(grep -o '__[A-Z][A-Z0-9_]*__' "$RENDER" | sort -u | tr '\n' ' ')"
  UNIT_PUT="$(unit_put "$RENDER" "$INSTALLED")"
  case "$UNIT_PUT" in
    same) echo "unit unchanged" ;;
    deps) sudo systemctl daemon-reload
          echo "unit updated: only the lines naming the other lanes changed, which the lane does not read" ;;
    *)    sudo systemctl daemon-reload
          echo "unit installed at $INSTALLED" ;;
  esac
  # A lane that serves goes on serving the unit, the runtime and the start scripts it started with:
  # said here when no proof follows to restart it.
  LANE_FILES=("$INSTALLED" "$CONFIG_DIR/vllm-auth/vllm_engine_auth.py" "$CONFIG_DIR/engine-preflight.sh" "$CONFIG_DIR/engine-secrets.sh")
  if [ "$SMOKE" -eq 0 ] && lane_serves_older "$UNIT" "${LANE_FILES[@]}"; then
    echo "NOTE: Nemotron runs older files than the ones installed now (its unit, its runtime or its start scripts); it runs these after: sudo systemctl restart $UNIT   (or Load it again from the cockpit)"
  fi
  # Images of this lane built from older pins, once the unit names this one.
  for img in $(docker images --format '{{.Repository}}:{{.Tag}}' 'qwen38-nemotron' 2>/dev/null); do
    [ "$img" = "$LOCAL_IMAGE" ] || { docker rmi "$img" >/dev/null 2>&1 && echo "removed the older build $img"; }
  done
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Nemotron 3.5 Bolt in the Lanes view"
  echo "  a terminal : ./switch-model.sh nemotron, then the commands it prints"
}

step "1/5 Preflight (Bolt: $MODEL @ ${MODEL_REV:0:12}, drafter $DRAFT @ ${DRAFT_REV:0:12})"
case "$(uname -m)" in aarch64|arm64) : ;; *) die "this lane is verified on the DGX Spark's ARM64 GB10 only (this box is $(uname -m))." ;; esac
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this needs the NVIDIA driver."
command -v docker >/dev/null || die "docker is required (stock on DGX OS)."
docker info >/dev/null 2>&1 || die "docker does not answer for $(id -un). Is it running, and is $(id -un) in the docker group?"
[ -s "$CONFIG_DIR/api-key" ] || die "no API key at $CONFIG_DIR/api-key. Run ./install.sh first: this lane serves behind the proxy with that key."
[ -f "$HERE/engine-preflight.sh" ] && [ -f "$HERE/engine-secrets.sh" ] && [ -f "$HERE/vllm_engine_auth.py" ] \
  && [ -f "$HERE/ckpt-fetch.py" ] \
  || die "engine-preflight.sh, engine-secrets.sh, vllm_engine_auth.py or ckpt-fetch.py is missing next to this script: run it from the repository."
# Measured where each part lands: the weights go to HF_CACHE, the image to Docker's own
# disk. What the cache already holds of these checkpoints comes off their share, by bytes.
existing(){ local p="$1"; while [ ! -e "$p" ]; do p="$(dirname "$p")"; done; printf '%s\n' "$p"; }
free_gb(){ { df -BG --output=avail "$(existing "$1")" 2>/dev/null || true; } | tail -1 | tr -dc '0-9'; }
HAVE_B=0
for R in "$MODEL" "$DRAFT"; do
  B="$({ du -s --apparent-size -B1 -L "$HF_CACHE/hub/models--${R//\//--}/blobs" 2>/dev/null || true; } | cut -f1)"
  HAVE_B=$(( HAVE_B + ${B:-0} ))
done
WEIGHTS_NEED=$(( (WEIGHTS_GB * 1000000000 - HAVE_B) / 1000000000 )); [ "$WEIGHTS_NEED" -ge 0 ] || WEIGHTS_NEED=0
# --prepare: the checkpoints are the Load's switch's to fetch, and to measure
[ "$MODE" = prepare ] && WEIGHTS_NEED=0
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
IMAGE_NEED="$IMAGE_GB"
docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1 && IMAGE_NEED=0
docker image inspect "$NEMOTRON_BASE_IMAGE" >/dev/null 2>&1 && IMAGE_NEED=0
FREE_W="$(free_gb "$HF_CACHE")"; FREE_D="$(free_gb "$DOCKER_ROOT")"
# -L: a cache moved to another disk behind a symlink (~/.cache/huggingface on an SSD) is on
# that disk; the link's own device is the home directory's.
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the checkpoints come with the lane's first Load"
  install_service
  step "Done: the Nemotron lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Nemotron 3.5 Bolt in the Lanes view builds its image (about 24 GB, 10 to download),"
  echo "               fetches its checkpoints (about $WEIGHTS_GB GB), then starts it"
  echo "  a terminal : ./install-nemotron.sh --prepare, then ./switch-model.sh nemotron and the commands it prints"
  exit 0
elif [ "$(stat -L -c %d "$(existing "$HF_CACHE")")" = "$(stat -L -c %d "$(existing "$DOCKER_ROOT")")" ]; then
  WANT=$((WEIGHTS_NEED + IMAGE_NEED))
  [ "${FREE_W:-0}" -ge "$WANT" ] \
    || die "${FREE_W:-?} GB free on the disk of $HF_CACHE and $DOCKER_ROOT, this needs about $WANT GB ($WEIGHTS_NEED for the checkpoints, $IMAGE_NEED for the image). Free some space, or ./install.sh --no-nemotron to leave this lane out."
else
  [ "${FREE_W:-0}" -ge "$WEIGHTS_NEED" ] \
    || die "${FREE_W:-?} GB free under HF_CACHE=$HF_CACHE, the checkpoints need about $WEIGHTS_NEED GB more. Free some space, or ./install.sh --no-nemotron."
  [ "${FREE_D:-0}" -ge "$IMAGE_NEED" ] \
    || die "${FREE_D:-?} GB free under $DOCKER_ROOT, the image needs about $IMAGE_NEED GB. Free some space, or ./install.sh --no-nemotron."
fi
echo "OK: $(uname -m), key present, checkpoints $([ "$MODE" = prepare ] && echo "left to the Load's switch" || { [ "$WEIGHTS_NEED" -eq 0 ] && echo 'already cached'; } || echo "$WEIGHTS_NEED GB to download") into $HF_CACHE, engine on $ENGINE_BIND:$PORT"

step "2/5 Image ($LOCAL_IMAGE)"
if docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
  echo "already built"
else
  docker image inspect "$NEMOTRON_BASE_IMAGE" >/dev/null 2>&1 || docker pull "$NEMOTRON_BASE_IMAGE" \
    || die "docker pull of $NEMOTRON_BASE_IMAGE failed. Causes: no internet, a Docker Hub rate limit (retry in a few minutes, or docker login), or the digest removed upstream."
  BUILD="$(mktemp -d)"
  curl -fsSL --retry 3 -o "$BUILD/$B12X_WHEEL" "$NEMOTRON_B12X_URL" \
    || { rm -rf "$BUILD"; die "could not download the b12x wheel from PyPI ($NEMOTRON_B12X_URL)"; }
  echo "$NEMOTRON_B12X_SHA256  $BUILD/$B12X_WHEEL" | sha256sum -c --quiet - \
    || { rm -rf "$BUILD"; die "the b12x wheel does not match the sha256 PyPI published for $NEMOTRON_B12X_VERSION; refusing to build."; }
  # The base image plus the wheel with --no-deps, so pip never replaces the vLLM, torch, CUTLASS
  # DSL and transformers the base carries; the last line proves the version installed.
  cat > "$BUILD/Dockerfile" <<'DEOF'
ARG BASE_IMAGE
ARG B12X_VERSION
FROM ${BASE_IMAGE}
ARG B12X_VERSION
COPY *.whl /tmp/
RUN pip install --no-deps /tmp/*.whl && rm /tmp/*.whl \
 && python3 -c "import sys; from importlib.metadata import version; v = version('b12x'); assert v == sys.argv[1], v" "${B12X_VERSION}"
DEOF
  docker build -q --build-arg BASE_IMAGE="$NEMOTRON_BASE_IMAGE" --build-arg B12X_VERSION="$NEMOTRON_B12X_VERSION" \
    -t "$LOCAL_IMAGE" "$BUILD" >/dev/null \
    || { rm -rf "$BUILD"; die "the image build failed (the lines above say why)"; }
  rm -rf "$BUILD"
  echo "built from $NEMOTRON_BASE_IMAGE and the b12x wheel"
fi

if [ "$MODE" = prepare ]; then
  step "Done: the Nemotron lane's image is ready; its Load fetches the checkpoints and starts it"
  exit 0
fi

step "3/5 Checkpoints ($MODEL @ ${MODEL_REV:0:12} and its drafter, resumable)"
mkdir -p "$HF_CACHE"
DL_TOKEN_ARGS=()
# the byte count, when the script is there (a mount of a missing file makes a root-owned directory)
DL_PROGRESS_ARGS=()
[ -f "$HERE/hf-progress.py" ] && DL_PROGRESS_ARGS=(-v "$HERE/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
# by name: docker takes the value from its own environment, so the token is in no argv
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; DL_TOKEN_ARGS=(-e HF_TOKEN); fi
# The image's own huggingface_hub, as for the other text lanes, with the same Xet ban, through
# ckpt-fetch.py, the one fetch of every lane's checkpoint: --offline-first takes a snapshot the
# cache holds only when every shard its index names is there.
fetch_ckpt(){  # $1 repo, $2 revision
  docker run --rm --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
    -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 "${DL_TOKEN_ARGS[@]}" \
    -v "$HERE/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" "${DL_PROGRESS_ARGS[@]}" \
    -v "$HF_CACHE":/hf "$LOCAL_IMAGE" /opt/qwen38/ckpt-fetch.py "$1" "$2" --offline-first \
    || die "download of $1 failed. Causes: no internet, HuggingFace throttling an unauthenticated download (set HF_TOKEN=<your token>), a pinned revision removed upstream, or a permission error in $HF_CACHE. Re-running resumes."
}
fetch_ckpt "$MODEL" "$MODEL_REV"
fetch_ckpt "$DRAFT" "$DRAFT_REV"

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves (a reply and a tool call, then the box goes back to the lane it was serving)"
WAS=""
for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-llamacpp.service; do
  systemctl is-active --quiet "$u" 2>/dev/null && WAS="$u"
done
WAS_NEMOTRON=0
systemctl is-active --quiet "$UNIT" 2>/dev/null && WAS_NEMOTRON=1
[ -n "$WAS" ] && echo "note: $WAS is serving and will be stopped for this test, then started again."
# From here on the serving lane is down: putting it back cannot live on the happy path.
restore_lane() {
  [ "$WAS_NEMOTRON" -eq 1 ] || sudo systemctl stop "$UNIT" 2>/dev/null || true
  if [ -n "$WAS" ]; then
    echo "starting $WAS again"
    sudo systemctl start "$WAS" || echo "WARNING: could not restart $WAS. Do it by hand: sudo systemctl start $WAS"
  fi
  rm -f "$RENDER" "${OUT:-}"
}
trap restore_lane EXIT
KEY="$(head -1 "$CONFIG_DIR/api-key")"
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
if [ "$WAS_NEMOTRON" -eq 1 ] && lane_serves_older "$UNIT" "${LANE_FILES[@]}"; then
  echo "Nemotron runs older files than the ones installed now (its unit, its runtime or its start scripts): it restarts to prove these"
  sudo systemctl restart "$UNIT"
elif [ "$WAS_NEMOTRON" -eq 0 ]; then
  sudo systemctl start "$UNIT"
fi
# read after the start: a start or a restart resets the count
R0="$(systemctl show -p NRestarts --value "$UNIT" 2>/dev/null | tr -dc '0-9')"
READY=0
for i in $(seq 1 150); do
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { READY=1; break; }
  if [ "$(systemctl is-active "$UNIT" || true)" = failed ] \
     || [ "$(systemctl show -p NRestarts --value "$UNIT" 2>/dev/null | tr -dc '0-9')" -gt "${R0:-0}" ]; then
    { sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager || true; } | tail -40
    die "the lane died while starting; the journal above says why."
  fi
  [ $((i % 15)) -eq 0 ] && echo "  still loading... ($((i * 8)) s; about 8 min on the reference box)"
  sleep 8
done
# a journal that cannot be read (sudo without a ticket) must not end this before the reason
[ "$READY" -eq 1 ] || { { sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager || true; } | tail -40; die "the lane did not answer /health within 20 min. The journal above says why."; }
echo "up after ~$(( (i - 1) * 8 )) s"
OUT="$(mktemp)"
ask(){  # $1: request body; the reply lands in $OUT, the HTTP code is printed
  curl -sS -o "$OUT" -w '%{http_code}' -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H @<(printf 'Authorization: Bearer %s\n' "$KEY") -H 'Content-Type: application/json' -d "$1" || true
}
CODE="$(ask '{"model":"nemotron-3.5-bolt","messages":[{"role":"user","content":"Reply with exactly: READY"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}')"
[ "$CODE" = 200 ] || { echo "HTTP ${CODE:-000}: $(head -c 300 "$OUT")"; die "the lane came up and its first generation failed (HTTP ${CODE:-000})."; }
python3 - "$OUT" <<'PY' || die "the lane answered 200 without the expected reply."
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
text = (m.get("content") or "").strip()
if "READY" not in text:
    raise SystemExit(f"reply: {text[:200]!r}")
print(f"   reply: {text[:40]!r}")
PY
CODE="$(ask '{"model":"nemotron-3.5-bolt","messages":[{"role":"user","content":"What is the weather in Berlin? Use celsius."}],"max_tokens":2048,"chat_template_kwargs":{"enable_thinking":false},"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]}')"
[ "$CODE" = 200 ] || { echo "HTTP ${CODE:-000}: $(head -c 300 "$OUT")"; die "the tool-call request failed (HTTP ${CODE:-000})."; }
python3 - "$OUT" <<'PY' || die "the lane did not return a parsed tool call (the qwen3_coder parser is not in effect?)."
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
calls = m.get("tool_calls") or []
if not calls or calls[0]["function"]["name"] != "get_weather":
    raise SystemExit(f"no get_weather call: {json.dumps(m)[:300]}")
args = json.loads(calls[0]["function"]["arguments"])
print(f"   tool call: get_weather({args})")
PY
# The key must not be in the lane's journal (vLLM masks api_key in its argument log line).
# Read whole first, then searched: piped into grep -q under pipefail, a journal long enough
# to fill the pipe died of SIGPIPE at the first match, and the check read "not found".
JOURNAL_COPY="$(mktemp)"
# shellcheck disable=SC2024  # the copy is this user's own temporary file, written by this shell
if ! sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager > "$JOURNAL_COPY" 2>/dev/null; then
  rm -f "$JOURNAL_COPY"; die "the lane's journal could not be read to check the key is not in it."
fi
if grep -q -F -f <(head -1 "$CONFIG_DIR/api-key") "$JOURNAL_COPY"; then
  rm -f "$JOURNAL_COPY"; die "the API key appears in the lane's journal: rotate it, and report this."
fi
echo "   the key is in none of the lane's $(wc -l < "$JOURNAL_COPY") journal lines"
rm -f "$JOURNAL_COPY"
# the trap stops the lane and brings the serving lane back, whichever way this ends

step "Done: the Nemotron lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  switch to it : Load on Nemotron 3.5 Bolt in the cockpit, or ./switch-model.sh nemotron"
echo "  clients      : the proxy on :30001 with the box's key, model nemotron-3.5-bolt (OpenAI and Anthropic APIs)"
