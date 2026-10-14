#!/usr/bin/env bash
# The heavy half of a lane installed on demand, as the user, with no sudo: what the cockpit's
# first Load of the lane runs before its switch. The image and video lanes' runtime (a venv and
# SGLang Diffusion, the video lane's Turbo adapter) and Kolibri's image, each by its own
# installer's --prepare. The checkpoint comes with the Load's switch, as for any Load.
#
#   ./prepare-lane.sh image|video|kolibri
#
# The lane is an argument of its own: the cockpit's actions never build one argument out of a
# parameter (tests/test_cockpit_http.py holds them to that).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage(){ echo "usage: ./prepare-lane.sh <image|video|kolibri>" >&2; exit 2; }
[ "$#" -eq 1 ] || usage
case "$1" in
  image|video|kolibri) exec bash "$HERE/install-$1.sh" --prepare ;;
  *) usage ;;
esac
