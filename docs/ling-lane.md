# Ling 3.0 Flash MXFP4 Humming

This fork merges upstream v1.40.5 and preserves Nanbeige. Ling is an optional experimental
lane for the 128 GB Spark. It is not claimed hardware-validated by this change.

## Install and use

From the fork checkout as the regular Spark user, after installing the main stack:

```bash
./install.sh --with-ling
```

This builds the separate runtime, downloads the pinned checkpoint, installs the unit and
smoke-tests a reply and parsed tool call, then restores the previous serving engine.
Choose Load on Ling in the cockpit to keep serving it afterwards. Terminal users can run
`./switch-model.sh ling` and apply the systemctl commands it prints.
The shared API model name is `ling-v3-flash-fp4`.

`./install-ling.sh --units-only` installs without a runtime download; Load prepares it.
`./install-ling.sh --no-smoke` skips hardware proof, and must not be mistaken for validation.
`./install-ling.sh --uninstall` removes the lane, restores a bootable text lane when possible,
and retains model weights. `./install.sh --no-ling` skips this run's update.

## Pinned sources

- Model: `inclusionAI/Ling-3.0-flash-fp4` at `3bae1cf4011b48475b2cc038fff283af49053ebc`.
- SGLang fork: inclusionAI/sglang at `079d4046003daf757d8a2daa44a3bb8935174b5e`,
  from branch `ling_v3_support_mxfp4_humming`.
- CUDA/ARM Docker base: the stack's digest-pinned SGLang v0.5.22 image.

Recipe: https://github.com/inclusionAI/ling-cookbook/blob/main/guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming.ipynb
The source/model/base are pinned; dependency resolution while building the fork is not a
complete package lock. The image records its installed packages in `/opt/ling-runtime-packages.txt`.
The cookbook recommends Python 3.11. This container retains its pinned base's Python runtime;
compatibility of that combination must be checked by the first build and hardware smoke test.

## Optimisations and limits

The launch recipe follows the cookbook: Humming MoE, NVRTC compiler, online FP8 LM head,
NEXTN three steps/four draft tokens with the same pinned snapshot for target and draft,
FlashInfer attention, CUTLASS FP8 GEMM, shared-expert fusion disabled, 8K chunked prefill,
16K max prefill, 64-token pages, one running request and a static memory fraction of 0.68.
The notebook's 262,144-token YaRN settings are retained. They are publisher settings,
not locally established memory or quality ceilings. Do not increase the memory fraction
without measurements: the cookbook warns that 0.75 can cause driver lockups.

Online FP8 LM-head conversion and YaRN can affect outputs. They are enabled here as part
of the requested optimised recipe, not described as universally lossless.

Before the serving model starts, a separate container generates FlashInfer build rules,
stops its owned process group, verifies it is gone, then compiles SM121 kernels without
model weights resident. Compile jobs scale with available memory, capped at eight.
There is no global process killing. Cache volumes persist FlashInfer, Humming and Triton
artifacts. Start/stop cleanup removes the compile container on failed or cancelled starts.

Initial kernel generation and compilation are bounded; a failed build never marks the
lane ready. The current engine is stopped only when starting the Ling unit, as with other
exclusive lanes. Allow around 105 GB disk budget for weights and runtime, plus build layers.

The cockpit follows ready/health and lifecycle. Some throughput counters use the stack's
generic side-lane metrics reader and may be unavailable for this SGLang fork. Missing
counters are not a performance measurement. SystemOne remains Qwen-only: its labels and
calibration are not validated for Ling. Simultaneous resident models are not added here.

## Validation on your Spark

1. Run the full install without `--no-smoke`; inspect build and precompile logs.
2. Load Ling; verify model identity, key enforcement and streamed output through the proxy.
3. Exercise thinking on/off, a tool call and a follow-up tool-result turn.
4. Switch to Qwen/Nanbeige and back; verify reboot enablement and removal.
5. Warm up, discard the first batch, repeat three times, record medians, accept length,
   peak unified memory and the exact runtime package list before changing any limits.

The cookbook reports an improvement from 37.8 to approximately 53.8 tokens/s on its
Spark workload. This repository has not reproduced those numbers. Its sample streaming
cell reports a different rate; do not treat either as a guaranteed result for your prompts.
