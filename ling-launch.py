#!/usr/bin/env python3
"""Ling's official Spark Humming recipe, with the stack's bind/port and file-based key."""
import argparse
import os
from pathlib import Path
import sys

ENV = {
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0", "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_DSV4_FP4_DEQUANT": "0", "SGLANG_FP8_IGNORED_LAYERS": "",
    "SGLANG_ENABLE_FP8_LM_HEAD": "1", "HUMMING_COMPILER": "nvrtc",
    "HUMMING_CACHE_DIR": "/root/.humming/cache",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
    "HF_HUB_OFFLINE": "1", "DO_NOT_TRACK": "1",
}
ARGS = ['--model-path', '/root/.cache/huggingface/hub/models--inclusionAI--Ling-3.0-flash-fp4/snapshots/3bae1cf4011b48475b2cc038fff283af49053ebc', '--served-model-name', 'ling-v3-flash-fp4', '--trust-remote-code', '--dtype', 'bfloat16', '--tp-size', '1', '--ep-size', '1', '--host', '127.0.0.1', '--port', '30000', '--max-running-requests', '1', '--max-mamba-cache-size', '64', '--chunked-prefill-size', '8192', '--max-prefill-tokens', '16384', '--page-size', '64', '--context-length', '262144', '--cuda-graph-backend-decode', 'full', '--cuda-graph-max-bs-decode', '1', '--cuda-graph-bs-decode', '1', '--cuda-graph-backend-prefill', 'disabled', '--random-seed', '308534008', '--reasoning-parser', 'deepseek-r1', '--tool-call-parser', 'qwen25', '--attention-backend', 'flashinfer', '--disable-flashinfer-autotune', '--mem-fraction-static', '0.68', '--fp8-gemm-backend', 'cutlass', '--moe-runner-backend', 'humming', '--flashinfer-mxfp4-moe-precision', 'default', '--disable-shared-experts-fusion', '--speculative-algorithm', 'NEXTN', '--speculative-draft-model-path', '/root/.cache/huggingface/hub/models--inclusionAI--Ling-3.0-flash-fp4/snapshots/3bae1cf4011b48475b2cc038fff283af49053ebc', '--speculative-num-steps', '3', '--speculative-eagle-topk', '1', '--speculative-num-draft-tokens', '4', '--json-model-override-args', '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}', '--config', '/out/engine-secrets.yaml', '--enable-metrics', '--sleep-on-idle']

def command():
    args = list(ARGS)
    for flag, env, default in (("--host", "LING_BIND", "127.0.0.1"), ("--port", "LING_PORT", "30000")):
        args[args.index(flag) + 1] = os.environ.get(env, default)
    return [sys.executable, "-m", "sglang.launch_server", *args]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="inclusionAI/Ling-3.0-flash-fp4")
    parser.add_argument("--revision", default="3bae1cf4011b48475b2cc038fff283af49053ebc")
    opts = parser.parse_args()
    if opts.model_path != "inclusionAI/Ling-3.0-flash-fp4" or opts.revision != "3bae1cf4011b48475b2cc038fff283af49053ebc":
        sys.exit("Unit and built Ling runtime pins differ; rebuild this lane.")
    os.environ.update(ENV)
    model = Path(ARGS[ARGS.index("--model-path") + 1])
    if not (model / "config.json").is_file():
        sys.exit("Pinned Ling snapshot is missing; run the lane download first.")
    if not Path("/out/engine-secrets.yaml").is_file():
        sys.exit("Engine key configuration is missing.")
    os.execv(sys.executable, command())


if __name__ == "__main__":
    main()
