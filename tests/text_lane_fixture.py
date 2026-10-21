"""Text lanes that can start, for the tests of the boot a side lane's uninstall gives back (lanes.sh
text_lane_back, text_lane_give_back): the image their unit or launcher runs is on the box, and the
checkpoint it serves is whole in the cache it mounts (text_lane_prepared). The empty units these tests
wrote before name neither: since v1.37.7 they are lanes that cannot start, and get no boot (review of
2026-10-21)."""
import pathlib

IMAGE = "lmsysorg/sglang@sha256:" + "e" * 64
# the docker of a box that has that image and no other: a stub's first line, before the rest of it
DOCKER_HAS_IT = f'[ "$1 $2" = "image inspect" ] && {{ [ "$3" = "{IMAGE}" ]; exit; }}\n'


def _engine(hf, repo):
    return (f"  -v {hf}:/root/.cache/huggingface \\\n  {IMAGE} \\\n"
            f"  python3 -m sglang.launch_server \\\n    --model-path {repo} --tp-size 1\n")


def _snapshot(hf, repo):
    snap = pathlib.Path(hf) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "main"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "model.safetensors").write_text("w")


def put_ready(units, cfg, hf, lanes):
    """The units of these text lanes ("27b", "flash"), each one able to start: the 27B unit names the
    image and its checkpoint, the flash unit runs a launcher in cfg that does, and both checkpoints are
    whole in hf."""
    units, cfg = pathlib.Path(units), pathlib.Path(cfg)
    if "27b" in lanes:
        (units / "qwen38-sglang.service").write_text(
            "[Service]\nExecStart=/usr/bin/docker run --rm \\\n" + _engine(hf, "org/text-27b"))
        _snapshot(hf, "org/text-27b")
    if "flash" in lanes:
        (units / "qwen38-flash.service").write_text(f"[Service]\nExecStart=/bin/bash {cfg}/launch-flash.sh\n")
        (cfg / "launch-flash.sh").write_text("exec /usr/bin/docker run --rm \\\n" + _engine(hf, "org/text-flash"))
        _snapshot(hf, "org/text-flash")
