# Groot 65B A6B NVFP4 (experimental)

Groot replaces the retired Ling lane in this fork. The 128 GB Spark cockpit manages its
Prepare, Download, Load, Stop, logs and model library through the same exclusive-engine lifecycle.
A normal `./install.sh` installs the unit on demand. Load on **Groot 65B A6B** prepares the
shared image and downloads the checkpoint. `./install.sh --with-groot` also downloads and
smoke-tests it during installation. CLI: `./prepare-lane.sh groot`, then
`./switch-model.sh groot` and the start command it prints. API alias: `groot-65b-a6b`.

## Pins and runtime

- [Official NVFP4 checkpoint](https://huggingface.co/GrootTeam/Groot-65B-A6B-261023-NVFP4),
  revision `68a352a55f51590ce0d7741a5d001d0555b34d09`.
- Shared ARM64 GB10 vLLM/b12x image, digest from `install-flash-local.sh`.
- Standard compressed-tensors loader, checkpoint chat template copied unchanged.
  This is not NVIDIA QAD/modelopt format; no `modelopt_mixed` or `--load-format b12x` override.

The card lists 98.6 GB of weights: NVFP4 experts, FP8 n-gram embeddings and remaining layers
in BF16. This is a Qwen4Exp hybrid MoE with 65B main parameters, 6B active, plus a 51B-entry
parameter n-gram table and MTP layer. Budget at least 100 GB for checkpoint download and
about 28 GB peak image storage if the shared image is absent. Model files and API keys
remain local and reusable across updates.

## Initial optimisation settings

128K native context (131,072), one sequence, 4K chunked prefill, prefix caching,
aligned Mamba cache, BF16 activations, automatic KV dtype, GPU memory fraction 0.85.
Docker memory/no-swap limit is 116 GiB. No YaRN, speculative decoding, online LM-head
conversion or guessed FP4 kernel override. These are starting settings, not measured
Spark performance or a guarantee the maximum context fits. The card recommends at least
128K for complex reasoning. Lower memory use or context if startup profiling fails.

Reasoning parser is `qwen3`; tool parser is the card's `qwen3_coder`. OpenCode/pi expose
low, medium, xhigh and thinking-off variants using the checkpoint template. The client
context budget is 100K, output 16K, leaving room under the server limit. Text-only is
intentional for this first integration; multimodal paths are not validated. SystemOne
remains unavailable on this lane.

## NVFP4 versus GGUF

The [official GGUF Q4_K_M release](https://huggingface.co/GrootTeam/Groot-65B-A6B-261023-GGUF)
is listed at 78.8 GB plus a 5.3 GB MTP companion, versus 98.6 GB for NVFP4. Both are listed
as fitting 128 GB Spark at 256K by the publisher. Smaller files do not establish faster
inference. NVFP4 is integrated first because it reuses this cockpit's ARM64 CUDA vLLM
lifecycle; a second llama.cpp runtime is not added without comparative hardware evidence.

## Hardware validation still required

The pinned image predates this checkpoint. Architecture support and compressed-tensors
FP8 n-gram loading must be verified on Spark; the model card's minimum version numbers
alone do not establish compatibility with this particular image. There is no automatic
fallback to different weights or runtimes. On failure inspect the journal and report the
runtime version, loader error and peak memory before changing pins.

Validate a full installation, model identity, authentication, streamed completion,
reasoning on/off, tool call plus tool-result continuation, repeated long pi sessions,
stop/load and a switch back to the existing Qwen lane. Record TTFT, tokens/s and peak
unified memory after warmup. Do not label the lane inference-validated until these pass.
`./install-groot.sh --uninstall` removes its unit/client entry and preserves weights.
