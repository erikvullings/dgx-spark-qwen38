"""The image and video lanes cannot write ~/.config/qwen38 (2026-10-16).

These two lanes have no key (the diffusion runtime has no --api-key) and take paths from
whoever reaches them: media sources, dump and output paths. They run as the box's user, who owns
~/.config/qwen38, where the serving key and the engine-preflight.sh every lane runs before it
starts both live, so a lane bound past loopback could overwrite either (found in review,
2026-10-16). The units now mount that directory read-only for the lane.

Read off the templates, and off engine-preflight.sh, which runs inside the same restrictions.
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNITS = ("qwen38-image.service.template", "qwen38-video.service.template")


def service_section(text):
    """The [Service] lines of a unit, continuation lines joined."""
    body = text.split("[Service]", 1)[1].split("\n[", 1)[0]
    return re.sub(r"\\\n\s*", " ", body).splitlines()


def values(lines, key):
    return [v for ln in lines if ln.startswith(key + "=") for v in ln.split("=", 1)[1].split()]


class SideLanesConfigReadOnly(unittest.TestCase):
    def test_the_config_directory_is_read_only(self):
        for name in UNITS:
            with self.subTest(unit=name):
                lines = service_section((REPO / name).read_text())
                self.assertIn("__HOME__/.config/qwen38", values(lines, "ReadOnlyPaths"))

    def test_the_serving_key_is_out_of_the_lanes_reach(self):
        """The lanes take media sources from their callers, local paths included, with this
        user's rights; they hold no key of their own to need (2026-10-16)."""
        for name in UNITS:
            with self.subTest(unit=name):
                lines = service_section((REPO / name).read_text())
                self.assertIn("-__HOME__/.config/qwen38/api-key", values(lines, "InaccessiblePaths"))
                self.assertNotIn("--api-key", " ".join(values(lines, "ExecStart")))

    def test_the_preflight_still_runs_under_them(self):
        """It is read and run from the read-only directory, and needs no write there: a unit
        whose ExecStartPre fails never starts."""
        for name in UNITS:
            with self.subTest(unit=name):
                pre = values(service_section((REPO / name).read_text()), "ExecStartPre")
                self.assertIn("__HOME__/.config/qwen38/engine-preflight.sh", pre)
        script = (REPO / "engine-preflight.sh").read_text()
        self.assertNotIn("api-key", script)
        self.assertNotRegex(script, r"\.config/qwen38\S*\s*$|>>?\s*\S*\.config/qwen38")


if __name__ == "__main__":
    unittest.main(verbosity=2)
