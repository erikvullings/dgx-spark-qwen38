# shellcheck shell=bash
# lanes.sh: what switch-model.sh and the installers share about the lanes. Sourced, never run;
# it needs CONFIG_DIR, and reads LANES_UNIT_DIR (the units' directory, /etc/systemd/system).
#
# The text lane the box comes back to. A switch to a side lane (image, video, Kolibri) disables
# the text lane's unit, so enablement cannot say which text lane was in use: it is written down
# here, one file, the SGLang text unit (the 27B's or the flash's) the box had at boot last. Until
# v1.29 each side lane kept a link to the lane it took the boot from, written by every switch,
# read by install.sh's six lines and by three uninstalls, each its own way: a round trip between
# side lanes made two links name each other, a side lane removed left another's link naming a
# unit that was gone, and a new lane meant editing all of it (found in review, 2026-10-14). With
# one record there is no chain to follow, and no loop to close.
LANES_UNIT_DIR="${LANES_UNIT_DIR:-/etc/systemd/system}"

text_lane_note(){  # $1 unit: the box's text lane is this one; anything but an SGLang text unit is ignored
  case "$1" in
    qwen38-sglang.service|qwen38-flash.service) mkdir -p "$CONFIG_DIR"; printf '%s\n' "$1" > "$CONFIG_DIR/text-lane" ;;
  esac
}

text_lane_last(){  # [$1 the side lane at boot] [$2 a file]: the text lane noted, else the one the links of v1.29 lead to
  # With $2, the record is not read, and only the links newer than that file are followed: the
  # record knows better than a link written before it (text_lane_migrate's second reading).
  local noted cur side ref="${2:-}"
  if [ -z "$ref" ]; then
    noted="$(cat "$CONFIG_DIR/text-lane" 2>/dev/null || true)"
    case "$noted" in qwen38-sglang.service|qwen38-flash.service) echo "$noted"; return 0 ;; esac
  fi
  # A box switched before v1.30 has only the links: they are followed from the side lane at boot
  # first (the others may be older than its switch), at most once through each side lane.
  for side in ${1:+"$1"} image video kolibri; do
    cur="qwen38-$side.service"
    for _ in 1 2 3 4; do               # its own link, then each other side lane's once
      case "$cur" in
        qwen38-sglang.service|qwen38-flash.service) echo "$cur"; return 0 ;;
        qwen38-image.service|qwen38-video.service|qwen38-kolibri.service)
          cur="${cur#qwen38-}"; cur="$CONFIG_DIR/lane-before-${cur%.service}"
          if [ -n "$ref" ] && ! [ "$cur" -nt "$ref" ]; then break; fi
          cur="$(cat "$cur" 2>/dev/null || true)" ;;
        *) break ;;
      esac
    done
  done
  return 0
}

text_lane_migrate(){  # once, on a box switched before v1.30: what its links say becomes the record
  local u="" side t f was="" ref=""
  if [ -s "$CONFIG_DIR/text-lane" ]; then
    # A link newer than the record was written after it by a version before v1.30 (the box
    # went back to one): it says what that version saw, so the links newer than the record are
    # read again, and the record answers where they lead to an older one (found in review,
    # 2026-10-14). v1.30 writes no link.
    for f in "$CONFIG_DIR/lane-before-image" "$CONFIG_DIR/lane-before-video" "$CONFIG_DIR/lane-before-kolibri"; do
      if [ -e "$f" ] && [ "$f" -nt "$CONFIG_DIR/text-lane" ]; then was="$(cat "$CONFIG_DIR/text-lane")"; fi
    done
    [ -n "$was" ] || return 0
    ref="$CONFIG_DIR/text-lane"
  fi
  # a text lane at boot is the text lane; else the links, from the side lane at boot first
  for t in qwen38-sglang.service qwen38-flash.service; do
    systemctl is-enabled --quiet "$t" 2>/dev/null && { u="$t"; break; }
  done
  if [ -z "$u" ]; then
    for side in image video kolibri; do
      if systemctl is-enabled --quiet "qwen38-$side.service" 2>/dev/null; then u="$(text_lane_last "$side" "$ref")"; break; fi
    done
  fi
  [ -n "$u" ] || u="$(text_lane_last "" "$ref")"
  [ -n "$u" ] || u="$was"              # links that lead nowhere new: the record stays what it said
  [ -z "$u" ] || text_lane_note "$u"
  return 0
}

text_lane_back(){  # [$1 the side lane at boot]: the text unit to give the boot back to, or nothing
  local u
  u="$(text_lane_last "${1:-}")"
  if [ -n "$u" ] && [ -f "$LANES_UNIT_DIR/$u" ]; then echo "$u"; return 0; fi
  # the one noted is gone (or nothing was noted): whichever text lane is installed, the 27B first
  for u in qwen38-sglang.service qwen38-flash.service; do
    [ -f "$LANES_UNIT_DIR/$u" ] && { echo "$u"; return 0; }
  done
  return 0
}

lane_serves_older_unit(){  # $1 unit, $2 its unit file: true when it serves and that file is newer than its start
  local started
  [ "$(systemctl show -p ActiveState --value "$1" 2>/dev/null)" = active ] || return 1
  started="$(systemctl show -p ExecMainStartTimestamp --value --timestamp=unix "$1" 2>/dev/null | tr -dc '0-9')"
  [ -n "$started" ] && [ -e "$2" ] && [ "$(stat -L -c %Y "$2")" -gt "$started" ]
}
