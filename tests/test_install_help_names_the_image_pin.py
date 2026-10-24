#!/usr/bin/env python3
"""The install help and the docker-pull advice name the pinned 27B image.

install.sh prints an "Env overrides" block that tells an operator which IMAGE to
pass, and the date that image was validated on the reference box. It also prints,
in the docker pull failure advice, the moving tag to try when the pinned digest is
gone. The IMAGE pin (its own line, the digest assignment with the
"= lmsysorg/sglang:vX.Y.Z, DATE" comment) is the source of truth for the version.
Two dates are not the same fact: the pin comment carries the date the image itself
was released upstream, while the help carries the later date it was validated and
pinned on the reference box, so the help date must not precede the pin date. The
help had drifted: it named v0.5.21 on 2026-10-09 and the pull advice named v0.5.21
while the pin is v0.5.22. This test reads all three straight from the file and
keeps them in step with the pin: the help and the advice carry the pin version,
and the help date is not earlier than the pin date. No network, no subprocess.
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install.sh").read_text()

# The "Env overrides" block is where the help lives; anchoring the date search
# there keeps it from ever reading the same phrase in the file header.
HELP_BLOCK = TEXT.split("Env overrides", 1)[1]


def image_pin():
    """The IMAGE pin line: the digest assignment and its "= lmsysorg/sglang:vX.Y.Z, DATE" comment."""
    match = re.search(
        r'(?m)^IMAGE="\$\{IMAGE:-lmsysorg/sglang@sha256:[0-9a-f]{64}\}"\s*#\s*=\s*'
        r'lmsysorg/sglang:(v[0-9.]+),\s*(\d{4}-\d{2}-\d{2})',
        TEXT,
    )
    assert match, "the IMAGE pin line with its version/date comment is missing"
    return match.group(1), match.group(2)


def help_version():
    """The IMAGE= tag in the Env overrides help block (the bare tag, not the digest pin)."""
    match = re.search(r'(?m)^\s*IMAGE=lmsysorg/sglang:(v[0-9.]+)\b', HELP_BLOCK)
    assert match, "the IMAGE= help line is missing"
    return match.group(1)


def help_date():
    """The validation date named in the Env overrides help block, anchored there on purpose."""
    match = re.search(r'(?m)the 27B image on (\d{4}-\d{2}-\d{2})', HELP_BLOCK)
    assert match, "the help does not name the 27B image validation date"
    return match.group(1)


def advice_version():
    """The IMAGE tag the docker pull failure advice tells an operator to try."""
    match = re.search(r'try IMAGE=lmsysorg/sglang:(v[0-9.]+) \./install\.sh', TEXT)
    assert match, "the docker pull advice does not name an IMAGE tag to try"
    return match.group(1)


class TestInstallHelpNamesTheImagePin(unittest.TestCase):
    def test_help_version_matches_pin(self):
        pin_version = image_pin()[0]
        self.assertEqual(
            help_version(), pin_version,
            f"help names {help_version()}, the IMAGE pin is {pin_version}",
        )

    def test_help_date_not_earlier_than_pin(self):
        pin_date = image_pin()[1]
        self.assertGreaterEqual(
            help_date(), pin_date,
            f"help date {help_date()} precedes the pin image date {pin_date}",
        )

    def test_pull_advice_version_matches_pin(self):
        pin_version = image_pin()[0]
        self.assertEqual(
            advice_version(), pin_version,
            f"pull advice names {advice_version()}, the IMAGE pin is {pin_version}",
        )


if __name__ == "__main__":
    unittest.main()
