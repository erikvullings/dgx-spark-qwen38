#!/usr/bin/env bash
# Install the Kolibri-1 lane: Aleph Alpha's open 78B mixture-of-experts (3.5B active per
# token), German and English, with reasoning and tool calling, served by vLLM on the text
# lanes' port behind the proxy and its key. Idempotent, and safe to run alone.
#
#   ./install-kolibri.sh                 first install, or update in place
#   ./install-kolibri.sh --no-smoke      skip the proof at the end
#   ./install-kolibri.sh --uninstall     remove the unit and the local image (weights kept)
#
# install.sh runs this on every plain install, like the other lanes; ./install.sh
# --no-kolibri leaves it out for good (a marker file remembers it). It takes 79 GB of
# weights (the official FP8 release) and a 22 GB image; when the disk does not have the
# room, this refuses before downloading anything and install.sh goes on without it.
#
# WHY vLLM AND NOT SGLANG. SGLang cannot load Kolibri (sgl-project/sglang#43256, open on
# 2026-10-14). Aleph Alpha serves it with vLLM and their plugin, aleph-alpha-inference 1.0.0,
# which pins vLLM 0.29.x. Their container is published for amd64 only, so this builds the
# same one here: their Dockerfile, on the very image their FROM line names (vLLM v0.29.0,
# whose arm64 manifest is pinned below by digest), plus the plugin's wheel, checked
# against the sha256 PyPI publishes. The -cu129 variant of that image cannot run on the
# GB10: its block-FP8 kernels are built for sm_120a, and the first GEMM dies with
# "cutlass_gemm_caller ... Error Internal" (measured 2026-10-14); this one is CUDA 13.0.
#
# TWO CHECKPOINTS, ONE LANE. The official FP8 is the default and the reference. A 4-bit
# NVFP4 variant (Quantinine/Kolibri-1-NVFP4A16-FP8: the routed experts in NVFP4, attention
# and everything else the official FP8 bit for bit) is the other one, switched to with
# ./switch-model.sh kolibri-nvfp4 and downloaded on that first switch. Measured on the
# reference box, 2026-10-14, against the FP8 over 153,450 positions of English, German and
# French Wikipedia: the likelihood of English and German text is level (+0.006 and +0.003
# nats per token, neither interval excludes zero), French drops measurably (+0.037 nats per
# token, about 3.8 % of perplexity; French is not one of Kolibri's two languages). It
# decodes 10 % faster alone and 50 % faster at 8 streams, in 42.8 GiB instead of 73.6.
#
# ONE ENGINE AT A TIME. The unit says so with Conflicts=: starting it stops whichever lane
# serves, and starting another lane stops it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-kolibri.service
INSTALLED="/etc/systemd/system/$UNIT"
CONTAINER=qwen38-kolibri
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
  # The two halves of a lane installed on demand, as for the image and video lanes:
  # --units-only puts in what needs root (the unit, its start scripts) and nothing heavy, and
  # --prepare builds the lane's image as the user, with no sudo, when the cockpit first loads
  # the lane. The checkpoint comes with that Load's switch, as for any Load.
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
  *) die "unknown option: $a" ;;
esac; done

CONFIG_DIR="$HOME/.config/qwen38"
# the text lane the box comes back to, one record shared with switch-model.sh (lanes.sh)
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"

# The pins, at column 0 like install.sh's: the vLLM image Aleph Alpha's Dockerfile builds on
# (v0.29.0, its arm64 manifest), the plugin's wheel, and the two checkpoints at the
# revisions measured on the reference box on 2026-10-14.
KOLIBRI_BASE_IMAGE="vllm/vllm-openai@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a"
KOLIBRI_PLUGIN_VERSION="1.0.0"
KOLIBRI_PLUGIN_URL="https://files.pythonhosted.org/packages/b0/d7/1eda35b6ee0293f2a52d7a653cf5a4d54c93f21e2e359fe70a7e7b87a851/aleph_alpha_inference-1.0.0-py3-none-any.whl"
KOLIBRI_PLUGIN_SHA256="5a0ca118e67924f10c4f9c04dd64bef117007a17dd820233a8841b9cb8d6f211"
KOLIBRI_FP8_REPO="Aleph-Alpha/Kolibri-1"
KOLIBRI_FP8_REV="258cae8ffffc8c12db670bdbc6ac01bcf8826c87"
KOLIBRI_NVFP4_REPO="Quantinine/Kolibri-1-NVFP4A16-FP8"
KOLIBRI_NVFP4_REV="4a8406455c474fd0fb2bd523c352cc99803eb492"
# Each checkpoint's share of the GPU memory and its download in GB (why: where the checkpoint is
# chosen below). switch-model.sh reads them here, with the repos and revisions.
KOLIBRI_FP8_GMU="0.80"
KOLIBRI_FP8_GB="79"
KOLIBRI_NVFP4_GMU="0.55"
KOLIBRI_NVFP4_GB="46"
# The local image: the base and the wheel, nothing else. Its tag names both, so a change of
# either pin is a new tag, and a box never serves an image built from older pins. The wheel
# is named by its version, which PyPI never lets a file change under (the tag said 1.0.0
# whatever the URL pinned, and a box that had it built kept it: found in review, 2026-10-14).
PLUGIN_WHEEL="aleph_alpha_inference-${KOLIBRI_PLUGIN_VERSION}-py3-none-any.whl"
[ "${KOLIBRI_PLUGIN_URL##*/}" = "$PLUGIN_WHEEL" ] \
  || die "KOLIBRI_PLUGIN_URL names ${KOLIBRI_PLUGIN_URL##*/}, not the $PLUGIN_WHEEL of KOLIBRI_PLUGIN_VERSION (repo bug: please open an issue)"
LOCAL_IMAGE="qwen38-kolibri:vllm0.29.0-aai${KOLIBRI_PLUGIN_VERSION}-${KOLIBRI_BASE_IMAGE##*:}"
LOCAL_IMAGE="${LOCAL_IMAGE:0:52}"

# The checkpoint: the installed unit's, or KOLIBRI_VARIANT (fp8, the default, or nvfp4).
INSTALLED_MODEL="$(grep -m1 -oE -- "(${KOLIBRI_FP8_REPO}|${KOLIBRI_NVFP4_REPO}) --revision" "$INSTALLED" 2>/dev/null | awk '{print $1}' || true)"
case "${KOLIBRI_VARIANT:-}" in
  "") VARIANT=fp8; [ "$INSTALLED_MODEL" = "$KOLIBRI_NVFP4_REPO" ] && VARIANT=nvfp4 ;;
  fp8|nvfp4) VARIANT="$KOLIBRI_VARIANT" ;;
  *) die "KOLIBRI_VARIANT takes fp8 or nvfp4, got: $KOLIBRI_VARIANT" ;;
esac
# Each checkpoint's share of the GPU memory, chosen so that both keep about the same KV cache,
# 2.0 to 2.2 million tokens (about 8 requests at the full 262,144 window): the FP8's 0.80 left
# 2,093,056 to 2,175,959 tokens over three boots, and MemAvailable never under 10.9 GiB
# (measured, the benchmark client on the box beside it); the NVFP4 holds 30.8 GiB less of
# weights, and 0.55 gives that back to the box rather than to a cache nobody fills.
if [ "$VARIANT" = nvfp4 ]; then
  MODEL="$KOLIBRI_NVFP4_REPO"; MODEL_REV="$KOLIBRI_NVFP4_REV"; GMU="$KOLIBRI_NVFP4_GMU"; WEIGHTS_GB="$KOLIBRI_NVFP4_GB"
else
  MODEL="$KOLIBRI_FP8_REPO"; MODEL_REV="$KOLIBRI_FP8_REV"; GMU="$KOLIBRI_FP8_GMU"; WEIGHTS_GB="$KOLIBRI_FP8_GB"
fi
# The base image's pull at its peak and the plugin's layer: 22.1 GB unpacked and 9.6 GB to
# download, which the containerd image store (Docker 29's default on a new install) keeps beside
# it, 29.5 GiB; ceil(x 1.02) + 1 as df -BG rounds up (install.sh counts the text lanes' images so).
# It asked for 23, the image alone (found in review, 2026-10-15).
IMAGE_GB=32

# The text lanes' cache and port: Kolibri is one of them for the proxy, so it answers where
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
PORT="${KOLIBRI_PORT:-${UPSTREAM_PORT:-$(text_flag --port)}}"; PORT="${PORT:-30000}"
ENGINE_BIND="${ENGINE_BIND:-$(text_flag --host)}"; ENGINE_BIND="${ENGINE_BIND:-127.0.0.1}"
case "$PORT" in ''|*[!0-9]*) die "the engine port must be a number, got: $PORT" ;; esac

if [ "$ACTION" = uninstall ]; then
  step "Removing the Kolibri lane"
  # what the box's links of v1.29 say, read while this lane is still the one at boot (lanes.sh)
  text_lane_migrate
  if [ -f "$INSTALLED" ]; then
    WAS_BOOT=0; systemctl is-enabled --quiet "$UNIT" 2>/dev/null && WAS_BOOT=1
    sudo systemctl disable --now "$UNIT" 2>/dev/null || true
    sudo rm -f "$INSTALLED"; sudo systemctl daemon-reload
    echo "unit removed"
    # Kolibri was the lane at boot: the text lane it replaced was disabled by that switch,
    # and removing this one would leave no engine at the next boot.
    if [ "$WAS_BOOT" -eq 1 ]; then
      # The text lane noted (lanes.sh), or the one installed: its own link, followed past a side
      # lane removed since, was v1.29's way (found in review, 2026-10-14), and one record needs none.
      BACK="$(text_lane_back kolibri)"
      if [ -n "$BACK" ]; then
        sudo systemctl enable "$BACK"
        echo "$BACK is enabled at boot again, as it was before Kolibri; start it now with: sudo systemctl start $BACK   (or the cockpit)"
      else
        echo "NOTE: no engine is enabled at boot any more. Start a lane from the cockpit, or:"
        echo "      sudo systemctl enable --now qwen38-sglang.service"
      fi
    fi
  fi
  # opencode and pi: the kolibri provider goes, and a default that named it follows the text
  # lane this box comes back to (lanes.sh; oc-point-default.py with no target: no name changes).
  # Left as they were, opencode offered and defaulted to a model nothing served, and its
  # small-model calls sent Kolibri's level "none" to the Qwen template, which refused it
  # (found in review, 2026-10-14).
  case "$(text_lane_back kolibri)" in
    qwen38-flash.service) OC_LANE=flash ;;
    *) OC_LANE=27b ;;
  esac
  OC_SUM_BEFORE="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
  for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
    [ -f "$OC_JSON" ] || continue
    WAS_DEFAULT=0
    grep -qsE '"model"[[:space:]]*:[[:space:]]*"kolibri/' "$OC_JSON" && WAS_DEFAULT=1
    python3 "$HERE/oc-merge-limits.py" "$OC_JSON" --remove-provider kolibri \
      || echo "NOTE: could not take the kolibri provider out of $OC_JSON; remove it by hand"
    if [ "$WAS_DEFAULT" -eq 1 ]; then
      python3 "$HERE/oc-point-default.py" "$OC_JSON" "$OC_LANE" "" "" \
        || echo "NOTE: could not point $OC_JSON at the $OC_LANE lane; set its \"model\" field yourself"
      # The compaction with it, from the window of the model put back: sized from Kolibri's, it
      # kept 46,000 tokens verbatim where a 27B at 1M keeps 132,000 (found in review, 2026-10-14,
      # at Kolibri's 38,000 then).
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
  rm -f "$CONFIG_DIR/lane-before-kolibri"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  for img in $(docker images --format '{{.Repository}}:{{.Tag}}' 'qwen38-kolibri' 2>/dev/null); do
    docker rmi "$img" >/dev/null 2>&1 && echo "image removed: $img"
  done
  # vLLM writes its compile cache as root inside the container: what the user cannot remove, sudo does
  rm -rf "$CONFIG_DIR/kolibri-cache" 2>/dev/null || sudo rm -rf --one-file-system "$CONFIG_DIR/kolibri-cache"
  rm -rf "$CONFIG_DIR/vllm-auth"
  echo "the base image $KOLIBRI_BASE_IMAGE stays (docker rmi it yourself if nothing else uses it)"
  for R in "$KOLIBRI_FP8_REPO" "$KOLIBRI_NVFP4_REPO"; do
    D="$HF_CACHE/hub/models--${R//\//--}"
    [ -d "$D" ] && echo "checkpoint kept, delete it yourself for the space: rm -rf $D"
  done
  exit 0
fi

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a
# lane installed on demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  mkdir -p "$CONFIG_DIR" "$CONFIG_DIR/kolibri-cache"
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
      -e "s|__IMAGE__|$LOCAL_IMAGE|g" -e "s|__MODEL__|$MODEL|g" -e "s|__REV__|$MODEL_REV|g" \
      -e "s|__GMU__|$GMU|g" -e "s|__HF_CACHE__|$HF_CACHE|g" \
      -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$RENDER"
  grep -q '__[A-Z][A-Z0-9_]*__' "$RENDER" && die "the unit template still holds an unsubstituted placeholder: $(grep -o '__[A-Z][A-Z0-9_]*__' "$RENDER" | sort -u | tr '\n' ' ')"
  if [ -f "$INSTALLED" ] && sudo cmp -s "$RENDER" "$INSTALLED"; then
    echo "unit unchanged"
  else
    sudo install -m 644 "$RENDER" "$INSTALLED"; sudo systemctl daemon-reload
    echo "unit installed at $INSTALLED"
  fi
  # A lane that serves goes on serving the unit, the runtime and the start scripts it started with:
  # said here when no proof follows to restart it, whether this run or an earlier update changed
  # them (an update leaves a serving lane serving, and said nothing for the image and video lanes:
  # found in review, 2026-10-14).
  LANE_FILES=("$INSTALLED" "$CONFIG_DIR/vllm-auth/vllm_engine_auth.py" "$CONFIG_DIR/engine-preflight.sh" "$CONFIG_DIR/engine-secrets.sh")
  if [ "$SMOKE" -eq 0 ] && lane_serves_older "$UNIT" "${LANE_FILES[@]}"; then
    echo "NOTE: Kolibri runs older files than the ones installed now (its unit, its runtime or its start scripts); it runs these after: sudo systemctl restart $UNIT   (or Load it again from the cockpit)"
  fi
  # Images of this lane built from older pins, once the unit names this one: removed before
  # the checkpoint came, a failed download left the unit naming an image that was gone.
  for img in $(docker images --format '{{.Repository}}:{{.Tag}}' 'qwen38-kolibri' 2>/dev/null); do
    [ "$img" = "$LOCAL_IMAGE" ] || { docker rmi "$img" >/dev/null 2>&1 && echo "removed the older build $img"; }
  done
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Kolibri-1 in the Lanes view"
  echo "  a terminal : ./switch-model.sh kolibri (or kolibri-nvfp4), then the commands it prints"
}

step "1/5 Preflight ($VARIANT: $MODEL @ ${MODEL_REV:0:12})"
case "$(uname -m)" in aarch64|arm64) : ;; *) die "this lane is verified on the DGX Spark's ARM64 GB10 only (this box is $(uname -m))." ;; esac
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this needs the NVIDIA driver."
command -v docker >/dev/null || die "docker is required (stock on DGX OS)."
docker info >/dev/null 2>&1 || die "docker does not answer for $(id -un). Is it running, and is $(id -un) in the docker group?"
[ -s "$CONFIG_DIR/api-key" ] || die "no API key at $CONFIG_DIR/api-key. Run ./install.sh first: this lane serves behind the proxy with that key."
[ -f "$HERE/engine-preflight.sh" ] && [ -f "$HERE/engine-secrets.sh" ] && [ -f "$HERE/vllm_engine_auth.py" ] \
  && [ -f "$HERE/ckpt-fetch.py" ] \
  || die "engine-preflight.sh, engine-secrets.sh, vllm_engine_auth.py or ckpt-fetch.py is missing next to this script: run it from the repository."
# Measured where each part lands: the weights go to HF_CACHE, the image to Docker's own
# disk. What the cache already holds of this checkpoint comes off its share, by bytes.
existing(){ local p="$1"; while [ ! -e "$p" ]; do p="$(dirname "$p")"; done; printf '%s\n' "$p"; }
free_gb(){ { df -BG --output=avail "$(existing "$1")" 2>/dev/null || true; } | tail -1 | tr -dc '0-9'; }
HAVE_B="$({ du -s --apparent-size -B1 -L "$HF_CACHE/hub/models--${MODEL//\//--}/blobs" 2>/dev/null || true; } | cut -f1)"
WEIGHTS_NEED=$(( (WEIGHTS_GB * 1000000000 - ${HAVE_B:-0}) / 1000000000 )); [ "$WEIGHTS_NEED" -ge 0 ] || WEIGHTS_NEED=0
# --prepare: the checkpoint is the Load's switch's to fetch, and to measure
[ "$MODE" = prepare ] && WEIGHTS_NEED=0
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
IMAGE_NEED="$IMAGE_GB"
docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1 && IMAGE_NEED=0
docker image inspect "$KOLIBRI_BASE_IMAGE" >/dev/null 2>&1 && IMAGE_NEED=0
FREE_W="$(free_gb "$HF_CACHE")"; FREE_D="$(free_gb "$DOCKER_ROOT")"
# -L: a cache moved to another disk behind a symlink (~/.cache/huggingface on an SSD) is on
# that disk; the link's own device is the home directory's, which skipped Docker's check.
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the checkpoint come with the lane's first Load"
  install_service
  step "Done: the Kolibri lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Kolibri-1 in the Lanes view builds its image (about $IMAGE_GB GB),"
  echo "               fetches its checkpoint (about $WEIGHTS_GB GB), then starts it"
  echo "  a terminal : ./install-kolibri.sh --prepare, then ./switch-model.sh kolibri and the commands it prints"
  exit 0
elif [ "$(stat -L -c %d "$(existing "$HF_CACHE")")" = "$(stat -L -c %d "$(existing "$DOCKER_ROOT")")" ]; then
  WANT=$((WEIGHTS_NEED + IMAGE_NEED))
  [ "${FREE_W:-0}" -ge "$WANT" ] \
    || die "${FREE_W:-?} GB free on the disk of $HF_CACHE and $DOCKER_ROOT, this needs about $WANT GB ($WEIGHTS_NEED for the checkpoint, $IMAGE_NEED for the image). Free some space, or ./install.sh --no-kolibri to leave this lane out."
else
  [ "${FREE_W:-0}" -ge "$WEIGHTS_NEED" ] \
    || die "${FREE_W:-?} GB free under HF_CACHE=$HF_CACHE, the checkpoint needs about $WEIGHTS_NEED GB more. Free some space, or ./install.sh --no-kolibri."
  [ "${FREE_D:-0}" -ge "$IMAGE_NEED" ] \
    || die "${FREE_D:-?} GB free under $DOCKER_ROOT, the image needs about $IMAGE_NEED GB. Free some space, or ./install.sh --no-kolibri."
fi
echo "OK: $(uname -m), key present, checkpoint $([ "$WEIGHTS_NEED" -eq 0 ] && echo 'already cached' || echo "$WEIGHTS_NEED GB to download") into $HF_CACHE, engine on $ENGINE_BIND:$PORT"

step "2/5 Image ($LOCAL_IMAGE)"
if docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
  echo "already built"
else
  docker image inspect "$KOLIBRI_BASE_IMAGE" >/dev/null 2>&1 || docker pull "$KOLIBRI_BASE_IMAGE" \
    || die "docker pull of $KOLIBRI_BASE_IMAGE failed. Causes: no internet, a Docker Hub rate limit (retry in a few minutes, or docker login), or the digest removed upstream."
  BUILD="$(mktemp -d)"
  curl -fsSL --retry 3 -o "$BUILD/$PLUGIN_WHEEL" "$KOLIBRI_PLUGIN_URL" \
    || { rm -rf "$BUILD"; die "could not download the aleph-alpha-inference wheel from PyPI ($KOLIBRI_PLUGIN_URL)"; }
  echo "$KOLIBRI_PLUGIN_SHA256  $BUILD/$PLUGIN_WHEEL" | sha256sum -c --quiet - \
    || { rm -rf "$BUILD"; die "the aleph-alpha-inference wheel does not match the sha256 PyPI published for $KOLIBRI_PLUGIN_VERSION; refusing to build."; }
  # Aleph Alpha's own Dockerfile (aleph-alpha-inference, 1.0.0): the base image plus the
  # wheel with --no-deps, so pip never replaces the vLLM, torch and transformers the base
  # carries. The last line proves vLLM will see the plugin: its entry point is how vLLM
  # finds it in every process it starts.
  cat > "$BUILD/Dockerfile" <<'DEOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY *.whl /tmp/
RUN pip install --no-deps /tmp/*.whl && rm /tmp/*.whl \
 && python3 -c "from importlib.metadata import entry_points; assert [e for e in entry_points(group='vllm.general_plugins') if e.name.startswith('aleph')], 'the plugin entry point is missing'"
DEOF
  docker build -q --build-arg BASE_IMAGE="$KOLIBRI_BASE_IMAGE" -t "$LOCAL_IMAGE" "$BUILD" >/dev/null \
    || { rm -rf "$BUILD"; die "the image build failed (the lines above say why)"; }
  rm -rf "$BUILD"
  echo "built from $KOLIBRI_BASE_IMAGE and the plugin's wheel"
fi

if [ "$MODE" = prepare ]; then
  step "Done: the Kolibri lane's image is ready; its Load fetches the checkpoint and starts it"
  exit 0
fi

step "3/5 Checkpoint ($MODEL @ ${MODEL_REV:0:12}, resumable)"
mkdir -p "$HF_CACHE"
DL_TOKEN_ARGS=()
# the byte count, when the script is there (a mount of a missing file makes a root-owned directory)
DL_PROGRESS_ARGS=()
[ -f "$HERE/hf-progress.py" ] && DL_PROGRESS_ARGS=(-v "$HERE/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
# by name: docker takes the value from its own environment, so the token is in no argv
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; DL_TOKEN_ARGS=(-e HF_TOKEN); fi
# The image's own huggingface_hub, as install.sh does for the other text lanes, with the
# same Xet ban (the classic CDN path measured 89 MB/s where Xet stalled at 0 to 8), through
# ckpt-fetch.py, the one fetch of every lane's checkpoint: --offline-first takes a snapshot
# the cache holds only when every shard its index names is there (huggingface_hub's own
# check sees a missing file only with its tree cache).
docker run --rm --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
  -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 "${DL_TOKEN_ARGS[@]}" \
  -v "$HERE/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" "${DL_PROGRESS_ARGS[@]}" \
  -v "$HF_CACHE":/hf "$LOCAL_IMAGE" /opt/qwen38/ckpt-fetch.py "$MODEL" "$MODEL_REV" --offline-first \
  || die "checkpoint download failed. Causes: no internet, HuggingFace throttling an unauthenticated download (set HF_TOKEN=<your token>), a pinned revision removed upstream, or a permission error in $HF_CACHE. Re-running resumes."

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves (a reply and a tool call, then the box goes back to the lane it was serving)"
WAS=""
for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-llamacpp.service; do
  systemctl is-active --quiet "$u" 2>/dev/null && WAS="$u"
done
WAS_KOLIBRI=0
systemctl is-active --quiet "$UNIT" 2>/dev/null && WAS_KOLIBRI=1
[ -n "$WAS" ] && echo "note: $WAS is serving and will be stopped for this test, then started again."
# From here on the serving lane is down: putting it back cannot live on the happy path.
restore_lane() {
  [ "$WAS_KOLIBRI" -eq 1 ] || sudo systemctl stop "$UNIT" 2>/dev/null || true
  if [ -n "$WAS" ]; then
    echo "starting $WAS again"
    sudo systemctl start "$WAS" || echo "WARNING: could not restart $WAS. Do it by hand: sudo systemctl start $WAS"
  fi
  rm -f "$RENDER" "${OUT:-}"
}
trap restore_lane EXIT
KEY="$(head -1 "$CONFIG_DIR/api-key")"
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
# A lane that runs older files than the ones installed now restarts for the proof: the engine that
# runs proved nothing about what this run installed (found in review, 2026-10-14).
if [ "$WAS_KOLIBRI" -eq 1 ] && lane_serves_older "$UNIT" "${LANE_FILES[@]}"; then
  echo "Kolibri runs older files than the ones installed now (its unit, its runtime or its start scripts): it restarts to prove these"
  sudo systemctl restart "$UNIT"
elif [ "$WAS_KOLIBRI" -eq 0 ]; then
  sudo systemctl start "$UNIT"
fi
# read after the start: a start or a restart resets the count, and one read before it, from a
# unit that had restarted on its own before, let as many crashes of the new one go unseen (found
# in review, 2026-10-14)
R0="$(systemctl show -p NRestarts --value "$UNIT" 2>/dev/null | tr -dc '0-9')"
READY=0
for i in $(seq 1 150); do
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { READY=1; break; }
  if [ "$(systemctl is-active "$UNIT" || true)" = failed ] \
     || [ "$(systemctl show -p NRestarts --value "$UNIT" 2>/dev/null | tr -dc '0-9')" -gt "${R0:-0}" ]; then
    { sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager || true; } | tail -40
    die "the lane died while starting; the journal above says why."
  fi
  [ $((i % 15)) -eq 0 ] && echo "  still loading... ($((i * 8)) s; about 10 min on the reference box)"
  sleep 8
done
# a journal that cannot be read (sudo without a ticket) must not end this before the reason:
# under pipefail it did, with only sudo's own line printed (found in review, 2026-10-14)
[ "$READY" -eq 1 ] || { { sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager || true; } | tail -40; die "the lane did not answer /health within 20 min. The journal above says why."; }
echo "up after ~$(( (i - 1) * 8 )) s"
OUT="$(mktemp)"
ask(){  # $1: request body; the reply lands in $OUT, the HTTP code is printed
  curl -sS -o "$OUT" -w '%{http_code}' -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H @<(printf 'Authorization: Bearer %s\n' "$KEY") -H 'Content-Type: application/json' -d "$1" || true
}
CODE="$(ask '{"model":"kolibri-1","messages":[{"role":"user","content":"Reply with exactly: READY"}],"max_tokens":64,"chat_template_kwargs":{"reasoning_effort":"none"}}')"
[ "$CODE" = 200 ] || { echo "HTTP ${CODE:-000}: $(head -c 300 "$OUT")"; die "the lane came up and its first generation failed (HTTP ${CODE:-000})."; }
python3 - "$OUT" <<'PY' || die "the lane answered 200 without the expected reply."
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
text = (m.get("content") or "").strip()
if "READY" not in text:
    raise SystemExit(f"reply: {text[:200]!r}")
print(f"   reply: {text[:40]!r}")
PY
# At effort none: at high (the template's default), the kolibri1 reasoning parser cuts the
# reasoning at a <tool_call> the model writes inside it, and the call is missed, on 1.9 to
# 3.5 % of the replies to an English weather request like this one (Quantinine, both
# checkpoints); at none they measured every call parsed. A first install must not fail by chance.
CODE="$(ask '{"model":"kolibri-1","messages":[{"role":"user","content":"What is the weather in Berlin? Use celsius."}],"max_tokens":2048,"chat_template_kwargs":{"reasoning_effort":"none"},"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]}')"
[ "$CODE" = 200 ] || { echo "HTTP ${CODE:-000}: $(head -c 300 "$OUT")"; die "the tool-call request failed (HTTP ${CODE:-000})."; }
python3 - "$OUT" <<'PY' || die "the lane did not return a parsed tool call (the kolibri1 parsers are not in effect?)."
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

step "Done: the Kolibri lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  switch to it : Load on Kolibri-1 in the cockpit, or ./switch-model.sh kolibri (kolibri-nvfp4)"
echo "  clients      : the proxy on :30001 with the box's key, model kolibri-1 (OpenAI and Anthropic APIs)"
