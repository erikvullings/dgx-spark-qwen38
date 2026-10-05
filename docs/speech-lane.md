# A speech lane: what was measured, and the design it points to

Not shipped. This is the measurement record behind the speech lane on the roadmap
(discussion [#38](https://github.com/hasso5703/dgx-spark-qwen38/discussions/38)): speech
recognition (ASR) and speech synthesis (TTS) on the same box as the text lanes. Measured
on the reference box on 2026-10-05, with the flash lane serving its abliterated target and
nothing else loaded, llama.cpp at master `6c59c40` built for sm_121 (CUDA 13.0).

## The short version

- **ASR is ready.** Qwen3-ASR (Apache-2.0) runs in `llama-server` from ggml-org's GGUF,
  behind an OpenAI-compatible `/v1/audio/transcriptions`, on the GPU or on the CPU alone.
  On the CPU, the 0.6B transcribes French at 1.6 times real time under load on five of the
  ten small cores, and costs the flash lane nothing measurable.
- **TTS is the weak half.** Qwen3-TTS runs in llama.cpp only as a command (`llama-tts`),
  at half of real time, and its HTTP endpoint is a pull request not yet merged. Kokoro-82M
  (Apache-2.0) runs on the CPU at 2.4 times real time under the same load, but has one
  French voice.
- **The GPU is the wrong place beside flash.** With the flash lane serving long agent
  prompts, a new process could not even create its CUDA context (out of memory, with
  17.6 GiB of `MemAvailable`), and a speech load on the GPU took 45% of the flash lane's
  decode. On the CPU, on cores of its own, it took 0 to 4%.

The design this points to: a speech lane on the CPU only, on the ten Cortex-A725 cores
(0-4 and 10-14), which the engines' busy threads do not use (the CPU-pinning A/B of
discussion #11). About 2.7 GiB of host memory, no GPU memory, and no cost to the text lane
it sits beside, flash included.

## ASR: Qwen3-ASR in llama-server

GGUF from ggml-org (`Qwen3-ASR-0.6B-GGUF`, `Qwen3-ASR-1.7B-GGUF`), Q8_0 with its Q8_0
audio encoder (mmproj): 1.0 GB and 2.5 GB on disk. Test sets: 20 clips of the Multilingual
LibriSpeech French test split (291 s, CC-BY-4.0) and 10 of LibriSpeech test-clean (92 s).
WER is word error rate after lower-casing and stripping punctuation; RTF is processing time
over audio time, one request at a time (below 1 is faster than real time).

| model | where | memory | WER fr | WER en | RTF fr |
|---|---|---|---|---|---|
| 0.6B Q8_0 | GPU (`-np 4 -c 8192`) | 2.4 GiB GPU + 0.5 GiB host | 9.65% | 0.40% | 0.113 |
| 1.7B Q8_0 | GPU (`-np 4 -c 8192`) | 3.7 GiB GPU + 0.7 GiB host | 5.51% | 0.40% | 0.047 |
| 0.6B Q8_0 | CPU only, 20 cores | 2.1 GiB host | 9.65% | 0.40% | 0.149 |
| 1.7B Q8_0 | CPU only, 20 cores | 3.5 GiB host | 5.64% | 0.40% | 0.213 |

The context size is most of the GPU cost: the 0.6B took 5.0 GiB of GPU memory at
`-np 8 -c 32768` against 2.4 at `-np 4 -c 8192`. `llama-server` returns the model's own
prefix with the text (`language French<asr_text>…`): a lane strips it, or the WER above
roughly doubles (17.3% raw for the 0.6B in French). For comparison, Whisper-large-v3 is
reported at 11.0% on the whole MLS French test, and Parakeet-TDT-0.6B-v3 at 6.1%
([arXiv:2509.14128](https://arxiv.org/abs/2509.14128)); twenty clips are a smaller sample
than either.

## TTS: Qwen3-TTS in llama.cpp, and Kokoro

| engine and model | memory | speed | round trip |
|---|---|---|---|
| `llama-tts`, Qwen3-TTS 1.7B Base Q8_0, default context | 7.2 GiB GPU + 2.2 GiB host | 0.44 x real time | 0.0% |
| same, `-c 2048` | 3.9 GiB GPU + 2.2 GiB host | 0.53 x real time | 0.0% |
| same, Q4_K_M, `-c 2048` | 3.3 GiB GPU + 2.1 GiB host | 0.50 x real time | 0.0% |
| Kokoro-82M ONNX on the CPU (`kokoro-onnx` 0.6.1) | 0.7 GiB host | 5.6 x real time (fr), 4.2 x (en) | 0.0% |

"Round trip" is the WER of the 1.7B ASR on the synthesized speech against the text it was
given (French, and English for Kokoro): every sample came back word for word. `llama-tts`
loads the model at every call (about 12 s), generates 5 to 6 frames per second where the
model's 12 Hz codec needs 12 for real time, and keeps one CPU core at 100% while the GPU
waits: each frame is 16 small sequential passes (the talker, then 15 code-predictor steps),
so the launches cost more than the arithmetic. Its server endpoint
([llama.cpp#26603](https://github.com/ggml-org/llama.cpp/pull/26603), streaming included)
and the CustomVoice variants (#29179) are open pull requests. Voice cloning there is a
speaker embedding only. The 0.6B needs a GGUF converted after
[#28231](https://github.com/ggml-org/llama.cpp/pull/28231). For reference, vllm-omni serves
Qwen3-TTS 0.6B at 0.25 RTF with 70 ms to the first audio, at about 8 GiB of GPU memory and
6.5 GiB of host memory for its two stages (a user's measurement beside the 27B, #38).

## What speech costs the text lane

`./bench.sh` greedy median on the flash lane, every run checked against the proxy's log for
agent traffic (runs that overlapped it are not in the table). Reference: 47.5 to 48.3 tok/s.

| speech running during the bench | greedy median |
|---|---|
| ASR 1.7B on the GPU, loaded and idle | 48.8 |
| ASR 1.7B on the GPU and `llama-tts`, both busy throughout | 26.4 (-45%) |
| ASR 0.6B on CPU cores 0-4 (5 threads) and Kokoro on 10-14 (5 threads), busy throughout | 47.5 (0%) |
| ASR 1.7B on cores 0-4 and Kokoro on 10-14, busy throughout | 46.3 (-2.5%) |
| ASR 1.7B on big cores 16-19 (4 threads) and Kokoro on 10-14, busy throughout | 45.7 (-3.8%) |

Under that load the speech side kept up as follows: the 0.6B on five small cores at RTF
0.63, the 1.7B at 1.31 on five small cores and 0.77 on four big ones, Kokoro at 0.41.
Unpinned and oversubscribed (ASR with 10 threads and Kokoro with all 20 on the same cores),
llama.cpp's ASR fell to 2.4 tokens per second, about ten times slower: the speech
processes need cores of their own, with no more threads than cores.

**The GPU beside flash.** After an hour of agent use with prompts of about 470 KB, the
flash scheduler held 96 GiB (98,039 MiB in `nvidia-smi`), and a new `llama-server` died in
`cudaSetDevice` with `out of memory` while `MemAvailable` read 17.6 GiB, minutes after the
same command had started fine. This is the page-cache and driver-refusal behaviour recorded in
BENCHMARKS.md (host memory vs prompt length on the flash lane): free memory on the host is
not memory the driver will hand to a new CUDA context. A speech lane that needs the GPU
cannot promise to start beside flash.

## The design this points to

- **CPU only, small cores.** `llama-server` for ASR with `CUDA_VISIBLE_DEVICES=` hidden,
  `-ngl 0 -t 5`, pinned to cores 0-4; the TTS engine pinned to 10-14 with five threads. No
  GPU memory, no CUDA context, so nothing a text lane does can stop it starting.
- **ASR: Qwen3-ASR 0.6B Q8_0 by default** (2.1 GiB, French WER 9.7% here, keeps up under
  load); the 1.7B as a choice for accuracy (3.5 GiB, French WER 5.6%) where the load is light.
- **TTS: Kokoro-82M** while llama.cpp's Qwen3-TTS is half of real time and has no server;
  Qwen3-TTS (ten languages, nine voices, cloning) once its endpoint lands and runs faster
  than real time, measured again then.
- **Served the way the other lanes are**: OpenAI-compatible routes (`/v1/audio/transcriptions`,
  `/v1/audio/speech`) on loopback behind the cockpit, a unit of its own, and the ASR prefix
  stripped before the answer leaves. Idle costs nothing: the ASR server measured 0% CPU
  between requests.

Open before it is built: the naturalness of the voices, which only listening tells; long
recordings (the clips here are 4 to 27 s); and a cockpit view, Load and proof like the
other lanes'.
