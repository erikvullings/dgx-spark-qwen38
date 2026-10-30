#!/usr/bin/env python3
"""The SGLang text engines open no port for their rendezvous, and keep gloo and NCCL on lo (2026-10-30).

An SGLang engine of one process still sets torch.distributed up, by default through a TCPStore at
tcp://127.0.0.1:<a free port>, and torch makes that store listen on every interface whatever address
it is given: the 27B lane, its container on the host's network, listened on *:23906 with no key on
the reference box, which has no firewall. Both pinned images read SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE
(srt/distributed/bootstrap.py, _resolve_dist_init_method; read in the 27B's v0.5.22 and the flash lane's
nightly on 2026-10-30): set to file:///tmp/sglang-rdv, the rendezvous goes through a file in the
container, which --rm makes new at each start, and no port opens for it. The 27B lane started that way
with nothing listening past loopback that FP4-Local alone did not have, and answered (2026-10-30).
"""
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAG = "-e SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=file:///tmp/sglang-rdv "
# gloo's and NCCL's bootstrap on lo: under the real clients the 27B lane listened on four ports of the
# reference box's LAN address without it, though the box's name resolves to 127.0.0.1 (2026-10-30), and
# the flash lane and the diffusion lanes did the same.
IFNAMES = "-e GLOO_SOCKET_IFNAME=lo -e NCCL_SOCKET_IFNAME=lo "


def sglang_starts():
    """Every file of the repo that starts an SGLang text engine."""
    return sorted(p.name for p in [*REPO.glob("*.template"), REPO / "run.sh"]
                  if "sglang.launch_server" in p.read_text())


class EveryEngineStartTakesTheFileRendezvous(unittest.TestCase):
    def test_the_starts_are_the_ones_this_test_knows(self):
        # a new SGLang lane is held to it as soon as its unit exists
        self.assertEqual(sglang_starts(), ["qwen38-flash-launch.sh.template", "qwen38-sglang-1m.service.template",
                                           "qwen38-sglang-64.service.template", "qwen38-sglang.service.template",
                                           "run.sh"])

    def test_each_docker_run_passes_it_once_with_a_fresh_container(self):
        for name in sglang_starts():
            with self.subTest(name=name):
                text = (REPO / name).read_text()
                run = text[text.index("docker run"):]
                self.assertEqual(run.count(FLAG), 1)
                self.assertEqual(run.count(IFNAMES), 1)
                # the file is the container's own: new at each start, never one a mount carries over
                self.assertIn("docker run --rm ", text)
                self.assertNotRegex(run, r"-v [^ ]*:/tmp[ /:]")

    def test_the_units_say_why(self):
        for name in sglang_starts():
            if name == "run.sh":
                continue                      # the native unit's docker run: CI compares their flags and
                                              # values, and the test above their -e lines
            with self.subTest(name=name):
                text = (REPO / name).read_text()
                self.assertIn("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE: the rendezvous of the engine's one process",
                              text[:text.index("docker run")])


if __name__ == "__main__":
    unittest.main()
