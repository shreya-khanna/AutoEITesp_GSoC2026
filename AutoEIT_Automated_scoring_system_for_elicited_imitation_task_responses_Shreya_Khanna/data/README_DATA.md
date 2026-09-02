# Data Directory — Spanish EIT Scoring

## Files

| File | Description | Size |
|---|---|---|
| `combined.xlsx` | Combined EIT responses with final scores — main training dataset (n=1,964) | ~55 KB |
| `idea_units_Spanish_AutoEIT.xlsx` | Rubric idea-unit breakdown per stimulus sentence | ~78 KB |
| `syllable-range_SpanishEIT.xlsx` | Stimulus syllable ranges (used during feature engineering development) | ~54 KB |
| `SpanishEIT_with_FinalScores_1.xlsx` | Sub-corpus 1 with rater scores per sheet | ~66 KB |
| `SpanishEIT_with_FinalScores_2.xlsx` | Sub-corpus 2 with rater scores per sheet | ~621 KB |
| `Copy of 038_EITReliability_Both Raters_*.xlsx` | Inter-rater reliability study sheets (participants 1-107) | various |

## Data Format

Each response sheet contains:
- `Sentence` — item number (1–33)
- `Stimulus` — the Spanish sentence learners heard (with syllable count in parentheses)
- `Transcription Rater 1` / `Transcription Rater 2` — orthographic transcriptions from two raters
- `Final Transcription` — agreed final transcription
- `Score Rater 1` / `Score Rater 2` — 0–4 ratings from each rater
- `Final Rating` / `Final Score` — final agreed score (99999 = requires adjudication)

## Provenance

This data was collected as part of the Spanish EIT reliability study at [institution]. Data was collected under IRB approval. The scoring rubric (idea units) was established in August 2025.

## Access

This dataset is **not publicly available** without authorization from the research team. If you would like access for research purposes, please contact:

- **PI:** [Contact name and email]
- **GSoC contributor:** Shreya Khanna

For reproducibility of the automated scoring pipeline, you will need:
1. `combined.xlsx` — your EIT responses with `stimulus`, `final transcription`, and optionally `final score` columns
2. `idea_units_Spanish_AutoEIT.xlsx` — the rubric idea units (can be substituted with your own rubric)

## Ethics

All data was collected with informed consent. Participant identifiers have been pseudonymized (e.g., `030.026`, `038.016` format). Do not re-identify participants.
