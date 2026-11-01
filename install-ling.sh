#!/usr/bin/env bash
# Ling was retired after repeated inference interruptions with pi.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = --uninstall ]; then exec bash "$HERE/retire-ling.sh"; fi
printf '%s\n' 'Ling is retired and is no longer available.' >&2
exit 1
