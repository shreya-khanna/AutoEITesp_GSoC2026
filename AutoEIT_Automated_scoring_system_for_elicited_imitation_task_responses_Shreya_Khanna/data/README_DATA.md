# Data Directory — Spanish EIT Scoring

## Files

| File | Description | Size |
|---|---|---|
| `combined.xlsx` | Combined EIT responses with final scores — main training dataset (n=1,964) | ~55 KB |
| `idea_units_Spanish_AutoEIT.xlsx` | Rubric idea-unit breakdown per stimulus sentence | ~78 KB |
| `syllable-range_SpanishEIT.xlsx` | Stimulus syllable ranges (used during feature engineering development) | ~54 KB |

## Data Format

The response sheet contains:
- `Sentence` — item number (1–33)
- `Stimulus` — the Spanish sentence learners heard (with syllable count in parentheses)
- `Transcription Rater 1` / `Transcription Rater 2` — orthographic transcriptions from two raters
- `Final Transcription` — agreed final transcription
- `Score Rater 1` / `Score Rater 2` — 0–4 ratings from each rater
- `Final Rating` / `Final Score` — final agreed score (99999 = requires adjudication)


For reproducibility of the automated scoring pipeline, you will need:
1. `combined.xlsx` — your EIT responses with `stimulus`, `final transcription`, and optionally `final score` columns
2. `idea_units_Spanish_AutoEIT.xlsx` — the rubric idea units (can be substituted with your own rubric)

## Ethics

All data was collected with informed consent. Do not re-identify participants.
