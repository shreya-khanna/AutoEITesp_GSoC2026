# AutoEIT — ASR for L2 Spanish Elicited Imitation

![Google Summer of Code x HumanAI Foundation](docs/images/blog_banner.png)
*Image credits: Google Summer of Code and HumanAI*

An automated ASR pipeline for **Elicited Imitation Test (EIT)** recordings of L2 (second-language) Spanish learners: audio segmentation, a twelve-model ASR transcription and evaluation harness, and a WER/MER/CER comparison against two independent human raters, measuring how closely automated transcription matches human transcription. (The downstream proficiency-scoring step that consumes these transcripts is a separate component, built by another student on the project.) Built as a [Google Summer of Code](https://summerofcode.withgoogle.com/) 2026 project with [HumanAI](https://humanai.foundation/).

**Start here:** [`docs/GSOC_FINAL_REPORT.md`](docs/GSOC_FINAL_REPORT.md) — goals, what was built, the headline results, and what's left to do.

## The problem

An Elicited Imitation Test (EIT) plays a learner a sentence and asks them to repeat it back; how
closely the repetition matches the original is a proxy for their language proficiency. Scoring that
at scale means transcribing thousands of short, disfluent, accented recordings automatically — and
verifying the transcription is actually trustworthy enough to build a proficiency score on top of,
rather than assuming it.

## What's in this repo

1. **Segmentation** (`src/preprocessing/`) — splits each raw, multi-item recording into per-sentence
   audio clips using Silero VAD.
2. **Transcription** (`src/transcription/`) — one script per ASR model: twelve systems total, spanning
   attention, CTC, and RNNT decoder families, plus three commercial APIs.
3. **Evaluation** (`src/postprocessing/`) — scores every model's output against two independent human
   raters on WER/MER/CER, and checks whether a model silently "corrects" a learner's genuine mistake
   back toward the scripted sentence — a failure mode aggregate WER can't see on its own.

## Headline result

| | WER | MER | CER | % perfect |
| --- | --- | --- | --- | --- |
| Human floor (rater vs. rater) | 0.161 | 0.150 | 0.076 | 42.9% |
| **`speechmatics`** (best overall) | 0.223 | 0.207 | 0.131 | 36.4% |
| **`canary`** (best open-source) | 0.248 | 0.213 | 0.134 | 36.8% |
| `whisper` (baseline) | 0.248 | 0.217 | 0.140 | 34.9% |

`speechmatics` and `canary` are this project's two recommendations — not only for accuracy, but
because they're also the least likely of the twelve models tried to silently revert a learner's
genuine mistake back to the scripted sentence, which would otherwise quietly undermine the whole
point of an Elicited Imitation Test. Full reasoning in the
[final report](docs/GSOC_FINAL_REPORT.md).

## Digging deeper

| Document | What's in it |
| --- | --- |
| [`docs/GSOC_FINAL_REPORT.md`](docs/GSOC_FINAL_REPORT.md) | The full GSoC summary: goals, results, best models, and lessons learned. |
| [`docs/asr_model_comparison_analysis.md`](docs/asr_model_comparison_analysis.md) | The detailed technical writeup — full ranking table, a section per model, and cross-cutting analyses. |
| [`notebooks/`](notebooks/) | One analysis notebook per finding, independently re-runnable. |

## Layout

```text
src/
  preprocessing/   VAD-based audio segmentation
  transcription/    one script per ASR model (12 models)
  postprocessing/    WER/MER/CER scoring, human-rater merging, review exports
notebooks/
  segmentation/        VAD segmentation, resegmentation, target-sentence localization
  transcription_quality/  early/baseline transcription exploration (pre-12-model)
  model_comparison/    per-model + cross-model WER/MER/CER comparison, human floor, stimulus correction
  plots/                saved plots (shared across all notebooks)
docs/                 final report, technical analysis doc, alternatives doc
scripts/              PBS batch scripts for HPC (Metis) runs
data/                 gitignored except data/model_review/ and small manifest
                      files — raw and segmented audio are never committed
```

## Reproducing a result

Every model follows the same four-step pattern; see Appendix B of `docs/asr_model_comparison_analysis.md` for the full commands.

## Data

Raw and segmented audio are never committed. `data/model_review/` (per-recording, per-model transcription exports) and a handful of small manifest files are tracked directly; the larger derived artifacts (human-rater sheets, full postprocessed scores, per-item ASR transcripts) are reproducible locally from the raw audio via the pipeline above but aren't committed, to keep the repository size manageable.
