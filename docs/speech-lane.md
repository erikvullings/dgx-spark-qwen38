# The speech lane: Qwen3-ASR and Kokoro, on the CPU, beside any lane

Speech recognition (ASR) and speech synthesis (TTS) on the same box as the text lanes,
behind OpenAI-compatible audio routes, on the CPU alone: no GPU memory, so it runs beside
whichever lane serves, flash included (discussion
[#38](https://github.com/hasso5703/dgx-spark-qwen38/discussions/38)). Qwen3-ASR hears 30
languages; Kokoro-82M speaks seven, with 41 voices. Opt-in, and not in the one command
yet: `./install-speech.sh` installs it, from a clone, on a box where `./install.sh` ran.

```bash
./install-speech.sh                    # Qwen3-ASR 0.6B and Kokoro-82M, proved before it ends
SPEECH_ASR=1.7b ./install-speech.sh    # the larger recognizer: French WER 5.6% against 9.7%, at real time
SPEECH_BOOT=1 ./install-speech.sh      # start it now, and with the box
./install-speech.sh --uninstall        # the unit and the runtime; the models stay
```

It needs `ffmpeg` (`sudo apt-get install -y ffmpeg`, which the video lane needs too), and
5 GB free where it installs (6 with the 1.7B), of which it uses about 2: llama.cpp's
`llama-server` built for the CPU at a pinned commit (318 MB, 2 min 20 s on the reference
box), a venv of pinned wheels (168 MB) and 1.4 GB of models checked by sha256 (2.9 GB with
the 1.7B). It writes `qwen38-speech.service`, then proves the
lane: it speaks a sentence and hears it back. The first run took 3 minutes here; a run on
an installed lane changes only what changed, and restarts the lane if it was running. The
one command's updates leave this lane as it is: run `./install-speech.sh` again after one,
to bring the lane to the version it installed.

It is not started at boot unless `SPEECH_BOOT=1`: running, it holds host memory that a
flash lane's long prompts also want, 2.5 GB resident at rest and 2.7 GB at work with the
0.6B (4.2 GB with the 1.7B), and no GPU memory at all. At rest it costs 0.01 of a core.
Start and stop it like any unit: `sudo systemctl start qwen38-speech.service`.

## Using it

Port 30024, on loopback and without a key, like the image and video lanes, but neither the
cockpit nor the proxy relays it: programs on the box reach it, and another machine through
an SSH tunnel (`ssh -N -L 30024:127.0.0.1:30024 <box>`). `SPEECH_BIND` binds another address
for someone who means to (the installer says what it costs). `SPEECH_PORT` moves it; the
recognizer takes the port after it, on loopback only.

```bash
# transcribe anything ffmpeg reads: a phone's m4a, a browser's webm or ogg, wav, mp3, flac
curl http://127.0.0.1:30024/v1/audio/transcriptions -F file=@meeting.m4a -F language=en
# speak: a WAV at 24 kHz ("response_format": "pcm" for the raw 16-bit samples)
curl http://127.0.0.1:30024/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Everything stays on this box.", "voice": "af_heart"}' -o hello.wav
# the voices, by language
curl http://127.0.0.1:30024/v1/voices
```

The OpenAI SDKs work as they are (openai-python 3.24.0, checked against the lane):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30024/v1", api_key="unused")
with open("meeting.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="qwen3-asr-0.6b", file=f).text)
client.audio.speech.create(model="kokoro-82m", voice="ff_siwis", input="Bonjour.",
                           response_format="wav").write_to_file("bonjour.wav")
```

- **Transcriptions** take a `file` of up to 100 MB, an optional `language`, passed on as a
  hint (it changed nothing here: the model names the language it hears, right in every test
  above), and `response_format` `json`, `text` or `verbose_json`, which adds the language
  heard and the duration; there are no segments or timestamps. `model` and `prompt` are
  accepted and ignored.
- **Speech** takes an `input` of up to 4,096 characters, a `voice` (OpenAI's names, `alloy`
  and the others, get the default voice, so clients that send them work), `speed` from 0.5
  to 2, `response_format` `wav` (the default) or `pcm`, and an optional `language` (else the
  voice's own). `mp3`, `opus`, `aac` and `flac` are refused with a 400 that says so. Open
  WebUI sends no format and converts what it gets, so it needs nothing.
- **One request at a time in each engine**: a second one waits its turn, and a short
  transcription takes its turn between two pieces of a long one.

## The languages

**Heard.** Qwen3-ASR recognizes 30 languages and 22 Chinese dialects, and identifies the
language itself ([its model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)). Measured
here on recorded speech in French and English (below), and on the lane's own speech in the
seven languages of the voices, where it named the language right each time, unasked.

**Spoken.** Kokoro-82M speaks seven languages with 41 voices, at about 0.4 of real time on
its five cores. Each language below said one sentence through the lane, and its recognizer
heard it back, on the reference box on 2026-10-05:

| language | voices | for example | heard back |
|---|---|---|---|
| American English | 20 | `af_heart` (the default) | word for word |
| British English | 8 | `bf_emma` | word for word |
| Spanish | 3 | `ef_dora` | word for word |
| French | 1 | `ff_siwis` | word for word |
| Italian | 2 | `if_sara` | word for word |
| Brazilian Portuguese | 3 | `pf_dora` | word for word |
| Hindi | 4 | `hf_alpha` | word for word |

Two languages Kokoro has are left out, and their voices refused rather than spoken wrong.
Japanese: espeak-ng, Kokoro's phonemizer here, reads every kanji out as "Chinese letter".
Mandarin: its voices were trained on misaki's pinyin phonemes and get espeak's instead
("zho" for 这, no tones); over four voices and four sentences, 21% of the characters came
back wrong through the 0.6B recognizer and 15% through the 1.7B, the same words both times
(这, 说, 会议). Both recognizers hear Chinese and Japanese: only speaking them is missing.
misaki's Chinese phonemizer needs jieba, which PyPI ships as source only, so it waits.

## Long recordings

One pass of the recognizer holds its context, the audio and the text together: 13 audio
tokens and 4 to 5 of text per second of speech, so the 8,192 tokens of the lane hold about
seven minutes. A longer recording sent whole came back cut, with a 200: 582 s of French
gave 353 words of 1,482. Long passes lose words before that, too. On the twenty French
clips end to end (291 s) and the ten English ones (92 s), heard in pieces of each length:

| pieces of | French, 291 s: WER | its time (x real time) | English, 92 s: WER |
|---|---|---|---|
| 20 s | 8.9% | 0.51 | 0.8% |
| **30 s (the lane's)** | **9.7%** | **0.51** | **0.4%** |
| 60 s | 9.5% | 0.53 | 0.4% |
| 120 s | 16.2% | 0.57 | 0.4% (one piece) |
| one pass | 9.0% | 0.78 | 0.4% |

At 120 s the recognizer skipped 61 words in the middle of a piece, right after a sentence
close to an earlier one, every time; up to 60 s it skipped nothing, and its error rate is
the one it has on the clips heard one by one (9.7%, above). So the routes never send a
recording whole: ffmpeg reads the upload as 16 kHz mono, it is cut into pieces of at most
30 s, each at the quietest tenth of a second of its last 7.5 s so that no word is cut in
two, and the pieces are heard one after the other and joined. A recording takes about half
of its own length: an hour of speech, about half an hour. Ten minutes of French (582 s)
came back whole that way, 1,513 words for the 1,482 of the reference. The 1.7B takes about
the length of the recording, at a WER of 6.3% end to end on the 291 s.

A box without ffmpeg (the installer asks for it) sends the upload as it came, in one pass:
WAV, MP3 or FLAC only, and a recording that fills the context is refused with a 413 rather
than returned cut.

## How it runs

One unit, two processes, the launcher starting both and ending both if either ends:

- **The recognizer**: Qwen3-ASR (Q8_0, with its Q8_0 audio encoder) in `llama-server`,
  `-ngl 0` with `CUDA_VISIBLE_DEVICES` empty, so that no CUDA context is ever made, five
  threads on CPUs 0-4, one request at a time, on loopback.
- **The routes and the voice**: `speech-server.py`, with Kokoro-82M in onnxruntime on five
  threads on CPUs 10-14. It reads every upload with ffmpeg, cuts it, sends the pieces to the
  recognizer, takes the model's `language French<asr_text>` prefix off its text, and speaks.
- **The small cores only**: the GB10's ten Cortex-A725 (CPUs 0-4 and 10-14), which the
  installer finds by their maximum frequency. The text engines' busy threads run on the ten
  Cortex-X925 (the CPU-pinning A/B of discussion #11). Each engine has cores of its own:
  sharing them made the recognizer ten times slower (below).
- **Memory**: ordinary host memory, which the unit's cgroup holds (`MemoryHigh` 4G and
  `MemoryMax` 5G with the 0.6B, 6G and 7G with the 1.7B), so that a runaway stops there
  and not at the box's memory edge. llama-server's prompt cache in host memory is off
  (`--cache-ram 0`): it keeps up to 8 GiB by default, which no recording hits twice, and it
  had taken the lane from 1.6 to 4 GiB in an afternoon.

## What was measured, and why the CPU

Measured on the reference box on 2026-10-05, with the flash lane serving its abliterated
target and nothing else loaded, llama.cpp at master `6c59c40` built for sm_121 (CUDA 13.0)
for the GPU rows and for the CPU alone for the others.


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

So the lane runs on the CPU only, on the ten Cortex-A725 cores (0-4 and 10-14), which the
engines' busy threads do not use (the CPU-pinning A/B of discussion #11): no GPU memory,
and no cost to the text lane it sits beside, flash included.

### ASR: Qwen3-ASR in llama-server

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

### TTS: Qwen3-TTS in llama.cpp, and Kokoro

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

### What speech costs the text lane

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

## Still open

- **Qwen3-TTS** (ten languages, nine voices, cloning), once llama.cpp's server endpoint
  ([#26603](https://github.com/ggml-org/llama.cpp/pull/26603)) lands and runs faster than
  real time, measured again then.
- **A cockpit view**: record and play in the browser, with Load and proof like the other
  lanes'.
- **The one command**: `./install.sh` does not run `./install-speech.sh` yet.
- **Japanese voices**, which need a phonemizer that reads kanji.
