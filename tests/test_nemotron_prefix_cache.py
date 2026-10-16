"""An agent's every turn resumes from the prefix cache on both Nemotron lanes (found 2026-10-16).

Bolt's unit followed its model card, which sets no prefix-match unit: its drafter's sliding-window
attention made vLLM align every prefix-cache hit to 17,984 tokens (the least common multiple of its
4,496-token attention block and 64), and the Mamba state is checkpointed one such unit before a
prompt's last one. An agent's conversation of 40,000 tokens recomputed 22,055 of them at every turn,
13.4 s before the first token, where the turn added 27. Super VL's card sets the fine-grained
matching NVIDIA wrote for it: 64-token blocks, hits at 64-token boundaries, and a Mamba checkpoint
where a request's prefix is shared. Both lanes carry it now, rendered in their units.
"""
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
FINE = ("--block-size 64", "--prefix-match-unit 64", "--enable-mamba-shared-prefix-checkpoint")


class NemotronLanesResumeFromThePrefixCache(unittest.TestCase):
    def test_each_nemotron_unit_matches_prefixes_at_64_tokens(self):
        for name in ("qwen38-nemotron.service.template", "qwen38-nemotron-super.service.template"):
            text = (REPO / name).read_text()
            command = text[text.index("ExecStart="):text.index("ExecStop=")]
            for flag in FINE:
                with self.subTest(unit=name, flag=flag):
                    self.assertEqual(command.count(flag), 1, f"{name}: {flag} once in its serving command")

    def test_no_nemotron_unit_turns_prefix_caching_off(self):
        # vLLM turns prefix caching on by default for these models (Mamba cache mode "align"); a
        # --no-enable-prefix-caching or a Mamba cache mode of "none" would recompute every prompt whole.
        for name in ("qwen38-nemotron.service.template", "qwen38-nemotron-super.service.template"):
            text = (REPO / name).read_text()
            for flag in ("--no-enable-prefix-caching", "--mamba-cache-mode none", "--mamba-cache-mode=none"):
                with self.subTest(unit=name, flag=flag):
                    self.assertNotIn(flag, text)


if __name__ == "__main__":
    unittest.main()
