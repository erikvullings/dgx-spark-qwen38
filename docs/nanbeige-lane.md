# Experimental Nanbeige4.3-33B lane

This fork adds an opt-in tenth engine lane. The checkpoint is
`primitive-ai/Nanbeige4.3-33B-mixed-NVFP4-FP8`, pinned at
`324d32dbaea09b2fbcde12d65943785778cb5a96`.
The original release is BF16, not FP16.

## Why this checkpoint

The [publisher's model card](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-mixed-NVFP4-FP8)
reports 25.30 GB versus 66.24 GB for BF16. Attention and eight sensitive MLP modules use FP8;
the remaining decoder linear layers use NVFP4; embeddings, the head and norms remain BF16.
It is compressed-tensors, not ModelOpt mixed precision. Quantization is detected from config,
so this lane must not inherit `--quantization modelopt_mixed` or `--load-format b12x`.

Primitive measured 48.0 tokens/s at concurrency one on an RTX PRO 6000 Blackwell (sm_120),
not the Spark's GB10 (sm_121). Its knowledge/tool-call scores were 88.9/79.0 versus
89.7/80.8 for BF16. These are publisher results, not measurements of this fork.
No Spark throughput, quality, maximum context or memory claim is made here.

## Install

Use the normal Spark user in the fork's checkout, after the main stack is installed:

```bash
./install.sh --with-nanbeige
```

This installs the unit, pulls the existing digest-pinned GB10 vLLM image shared with
Flash FP4-Local, downloads the checkpoint, smoke-tests a reply and parsed tool call,
and restores whichever engine was serving. Its compressed-tensors NVFP4/FP8 kernels
on GB10 still need validation. The smoke test failing means this lane is not validated;
do not interpret static checks as proof that the model runs.

For an already configured stack, install without the hardware smoke test only when desired:

```bash
./install-nanbeige.sh --no-smoke
```

Then Load Nanbeige from the cockpit. In a terminal, `./switch-model.sh nanbeige` stages the
switch and prints the systemctl commands to apply it, just like the other lanes.
The proxy exposes `nanbeige4.3-33b` on the usual API. The default window is conservatively
32,768 tokens, with two concurrent sequences and a memory fraction of 0.80. These are
initial experimental settings, not measured ceilings. The native model window of 512K
is not promised on this lane. KV cache stays at the engine's default precision.

About 27 GB is reserved as checkpoint download budget, plus the shared image (about 18 GB
on disk, if absent). Disk budget and live GPU allocation are different.

## Chat and tools

The original checkpoint chat template is copied without Qwen-specific patches.
The model card specifies `qwen3` reasoning and `qwen3_coder` tools. The OpenCode provider
is text-only and preserves thinking history for agent workflows. `none` and `minimal`
reasoning effort map to `enable_thinking=false`; other supported effort levels enable it.
This is a boolean control, not distinct model reasoning budgets.

## Validation on the Spark

1. Run `./install-nanbeige.sh` without `--no-smoke`. Check that it restores the prior lane.
2. Load Nanbeige in the cockpit and confirm its model name, ready state and logs.
3. Verify a streamed response, thinking on/off, a parsed tool call and a second tool-result turn.
4. Check API-key enforcement, a switch back to Qwen, then a switch back to Nanbeige.
5. Warm up, discard the first batch, repeat benchmarks three times, and record medians,
   process memory, context length and concurrency before raising any limit.

The existing `/v1/systemone` implementation remains SGLang/Qwen-only. Nanbeige does not
become a SystemOne model merely by adding this lane; its tokenizer labels and probabilities
would need separate validation.

## Remove or skip

`./install.sh --no-nanbeige` skips this run's update; it does not remove an installed lane.
`./install-nanbeige.sh --uninstall` removes it and restores a bootable text lane when possible;
checkpoint weights and shared images are retained, as in the other lane installers.

## Multiple resident models

This change retains exclusive engine switching. Keeping Nanbeige and Qwen resident together
requires separate ports, routing and combined KV/runtime budgets. The smaller checkpoint
helps that future change but does not implement it.
