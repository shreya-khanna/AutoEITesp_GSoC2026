# ASR model alternatives to compare against faster-whisper large-v3

Context: current pipeline (`src/transcription/transcribe.py`) uses faster-whisper `large-v3`
(`beam_size=1, temperature=0, condition_on_previous_text=False, word_timestamps=True`), with
language auto-detected in v1 and hardcoded to `"es"` in `transcribe_v2.py`. Primary goal is exact
transcription accuracy of short, per-sentence, L2-accented Spanish speech (EIT items, ~1-15s
each), to be scored against human reference transcriptions (also per-sentence) via WER / MER
(using `jiwer`, which computes both natively along with CER, WIL, WIP).

## Cheap to try — same faster-whisper pipeline, just swap config

| Option | Change | Why | Tradeoff |
|---|---|---|---|
| `large-v2` | `WhisperModel("large-v2", ...)` in `load_model()` | Anecdotally hallucinates less than v3 on short/noisy audio in several benchmarks | Slightly older model, may lag v3 on cleaner audio |
| `large-v3-turbo` | Same, swap model name | Faster (pruned decoder) | Likely small accuracy hit — speed/accuracy tradeoff, not a quality upgrade |
| `openai-whisper` (original repo) | Same weights, no CTranslate2 quantization | Sanity check that faster-whisper's int8/float16 quantization isn't the source of errors | Not really a different model — expect near-identical output |
| Decoding params on current model | `beam_size=5`, temperature fallback list (e.g. `[0, 0.2, 0.4, 0.6]`) instead of fixed `0`, optionally an `initial_prompt` biasing toward EIT stimulus vocabulary | Recovers accuracy on garbled/accented speech; plausible win since some errors look like language-model prior issues (e.g. "nieva" → "niega") not acoustic issues | More compute per item |

## Architecturally different — worth trying for a genuinely different failure mode

| Model | Type | Why | Tradeoff |
|---|---|---|---|
| `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` (via `transformers`) | CTC | Fails differently than Whisper's attention decoder — less prone to hallucinating fluent-sounding wrong text, more prone to literal phonetic garbling | No built-in word-level confidence like Whisper's per-word `prob`; would need to derive something from CTC logits — more work than a drop-in swap |
| `facebook/mms-1b-all` (Meta MMS, via `transformers`) | CTC | Trained on far more languages/accents, including more non-native speech patterns | Same confidence-score gap as above |
| SeamlessM4T v2 (Meta) | Attention-based, like Whisper | Strong massively multilingual alternative | Heavier to set up; second-tier candidate |

## Recommendation (as of 2026-07-05)

Don't spread thin across all of these at once. Start with `large-v2` as a free control (same
pipeline, one-line model-name change). If a genuinely different architecture is wanted in the
comparison, add one CTC-based model — Wav2Vec2-XLSR-Spanish is the more established/tested option
of the two CTC candidates. That gives three comparison points (current v3, v2, one CTC model)
without building three separate inference pipelines.

Wire this up once the human reference transcriptions are downloaded, so all model variants are
scored against the same ground truth in one pass.
