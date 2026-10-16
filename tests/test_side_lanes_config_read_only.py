"""The image and video lanes reach none of the box's secrets (2026-10-16, 2026-10-17).

These two lanes have no key (the diffusion runtime has no --api-key) and take paths from
whoever reaches them: media sources, dump and output paths. They run as the box's user, who owns
~/.config/qwen38, where the serving key (api-key, and the engine-secrets.yaml the text lanes
write at every start) and the engine-preflight.sh every lane runs before it starts all live, so
a lane bound past loopback could read the key or overwrite either (found in review, 2026-10-16).
The identity wall's client keys live in /etc/qwen38, readable by the same user (docs/clients.md).

A mask on each file is not enough (2026-10-17): systemd lays it on the path when the lane starts,
so a file created there later is not masked, and a rename over the masked path on the host, which
is how engine-secrets.sh writes the key, detaches the mask in the lane's namespace (measured with
a mount namespace on the reference box). So the lanes see ~/.config/qwen38 as an empty read-only
tmpfs with the preflight bound into it, and /etc/qwen38 not at all.

Read off the templates, and off engine-preflight.sh, which runs inside the same restrictions; the
names of the secret files are read off the script and the page that write them.
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNITS = ("qwen38-image.service.template", "qwen38-video.service.template")
CONFIG = "__HOME__/.config/qwen38"
PREFLIGHT = CONFIG + "/engine-preflight.sh"


def service_section(text):
    """The [Service] lines of a unit, continuation lines joined."""
    body = text.split("[Service]", 1)[1].split("\n[", 1)[0]
    return re.sub(r"\\\n\s*", " ", body).splitlines()


def values(lines, key):
    return [v for ln in lines if ln.startswith(key + "=") for v in ln.split("=", 1)[1].split()]


def under(path, top):
    return path == top or path.startswith(top.rstrip("/") + "/")


def bound(lines):
    """The paths bind-mounted into the lane's view, prefixes and options dropped."""
    return [b.lstrip("-").split(":")[0] for b in values(lines, "BindReadOnlyPaths") + values(lines, "BindPaths")]


def hidden(lines, path):
    """True when the lane's namespace hides path behind a directory: one InaccessiblePaths names,
    or a TemporaryFileSystem nothing binds the path back into. A mask on the file itself does not
    count (see above)."""
    if any(under(path, d.lstrip("-+")) and path != d.lstrip("-+") for d in values(lines, "InaccessiblePaths")):
        return True
    return any(under(path, t.split(":")[0]) and path != t.split(":")[0]
               and not any(under(path, b) for b in bound(lines))
               for t in values(lines, "TemporaryFileSystem"))


def secret_files():
    """Where the box keeps its keys, read off what writes them."""
    names = set(re.findall(r'"\$d/([\w.-]+)"', (REPO / "engine-secrets.sh").read_text()))
    wall = set(re.findall(r"(/etc/qwen38/[\w.-]+)", (REPO / "docs" / "clients.md").read_text()))
    return {CONFIG + "/" + n for n in names} | wall


class SideLanesConfigReadOnly(unittest.TestCase):
    def test_the_config_directory_is_read_only_and_shows_the_preflight_alone(self):
        for name in UNITS:
            with self.subTest(unit=name):
                lines = service_section((REPO / name).read_text())
                self.assertIn(CONFIG + ":ro", values(lines, "TemporaryFileSystem"))
                self.assertEqual([b for b in bound(lines) if under(b, CONFIG)], [PREFLIGHT])
                self.assertEqual([b for b in values(lines, "BindPaths") if under(b.lstrip("-"), CONFIG)], [],
                                 "a writable bind into the config directory")

    def test_no_secret_of_the_box_is_in_the_lanes_reach(self):
        """The serving key wherever the box writes it, and the wall's client keys (2026-10-17)."""
        secrets = secret_files()
        for must in ("api-key", "engine-secrets.yaml", "/etc/qwen38/client-keys.json"):
            self.assertTrue(any(s.endswith(must) for s in secrets), f"{must} not found where it is written")
        for name in UNITS:
            lines = service_section((REPO / name).read_text())
            self.assertNotIn("--api-key", " ".join(values(lines, "ExecStart")))
            for secret in sorted(secrets):
                with self.subTest(unit=name, secret=secret):
                    self.assertTrue(hidden(lines, secret), f"{secret} is in the lane's reach")
            with self.subTest(unit=name):
                # "-": a box without the wall has no /etc/qwen38, and the lane must start there
                self.assertIn("-/etc/qwen38", values(lines, "InaccessiblePaths"))

    def test_the_preflight_still_runs_under_them(self):
        """It is read and run from the read-only view, and needs nothing else there: a unit whose
        ExecStartPre fails never starts."""
        for name in UNITS:
            with self.subTest(unit=name):
                lines = service_section((REPO / name).read_text())
                self.assertIn(PREFLIGHT, values(lines, "ExecStartPre"))
                self.assertFalse(hidden(lines, PREFLIGHT))
        script = (REPO / "engine-preflight.sh").read_text()
        self.assertNotIn("api-key", script)
        self.assertNotIn(".config", script)
        self.assertNotRegex(script, r"\bdirname\b|BASH_SOURCE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
