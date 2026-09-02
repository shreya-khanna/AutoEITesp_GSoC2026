"""Evaluate ordinal scoring with optional LLM arbitration.

Only complete 30-sentence participant sheets with valid ``Final Rating`` or
``Final Score`` values are included. The script always reports the ordinal
baseline; pass ``--backend openai`` or ``--backend gemini`` to make live LLM
calls for low-confidence rows and compare the adjusted scores.

Examples:
    python evaluate_llm_arbitration.py
    python evaluate_llm_arbitration.py --backend openai
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

import score_responses as scoring


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "reliability_scored_test.xlsx"
MODEL = ROOT / "models" / "ordinal_svm_pipeline.pkl"
OUTPUT = ROOT / "models" / "llm_arbitration_evaluation.json"
EXPECTED_SENTENCES = 30


def load_model(path: Path):
    """Load current artifacts and older pickles serialized from __main__."""
    main_module = sys.modules["__main__"]
    setattr(main_module, "OrdinalThresholdEnsemble", scoring.OrdinalThresholdEnsemble)
    with path.open("rb") as handle:
        return pickle.load(handle)


def human_column(frame: pd.DataFrame) -> str | None:
    for column in ("Final Rating", "Final Score", "Score Rater 1"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.between(0, 4).sum() == EXPECTED_SENTENCES:
                return column
    return None


def metric_block(human: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "exact_sentence_agreement": float(accuracy_score(human, predicted)),
        "within_one_agreement": float(np.mean(np.abs(human - predicted) <= 1)),
        "qwk": float(cohen_kappa_score(human, predicted, weights="quadratic")),
        "mae": float(mean_absolute_error(human, predicted)),
    }


def evaluate(input_path: Path, model_path: Path, backend: str | None) -> dict:
    model = load_model(model_path)
    idea_units = scoring.load_idea_units(scoring.IDEA_UNITS_FILE)
    workbook = pd.ExcelFile(input_path)
    baseline_human: list[int] = []
    baseline_predicted: list[int] = []
    final_predicted: list[int] = []
    sheet_rows: list[dict] = []
    arbitrated_rows = 0
    complete_sheets = 0

    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)
        selected_human = human_column(frame)
        transcription_available = "Final Transcription" in frame.columns or "Transcription Rater 1" in frame.columns
        if selected_human is None or "Stimulus" not in frame.columns or not transcription_available:
            continue

        scored = scoring.score_dataframe(frame, model, idea_units, llm_backend=backend)
        human = pd.to_numeric(scored[selected_human], errors="coerce")
        baseline = pd.to_numeric(scored["AutoScore"], errors="coerce")
        final = pd.to_numeric(scored["Final_AutoScore"], errors="coerce")
        valid = human.between(0, 4) & baseline.between(0, 4) & final.between(0, 4)
        if valid.sum() != EXPECTED_SENTENCES:
            continue

        human_values = human[valid].astype(int).to_numpy()
        baseline_values = baseline[valid].astype(int).to_numpy()
        final_values = final[valid].astype(int).to_numpy()
        complete_sheets += 1
        sheet_arbitrated = int(scored.loc[valid, "Arbitrated"].sum())
        arbitrated_rows += sheet_arbitrated
        baseline_human.extend(human_values)
        baseline_predicted.extend(baseline_values)
        final_predicted.extend(final_values)
        sheet_rows.append({
            "sheet": sheet,
            "human_column": selected_human,
            "arbitrated_rows": sheet_arbitrated,
            "baseline_total": int(baseline_values.sum()),
            "final_total": int(final_values.sum()),
            "human_total": int(human_values.sum()),
            "baseline_total_difference": int(abs(baseline_values.sum() - human_values.sum())),
            "final_total_difference": int(abs(final_values.sum() - human_values.sum())),
            "baseline": metric_block(human_values, baseline_values),
            "with_arbitration": metric_block(human_values, final_values),
        })

    if not baseline_human:
        raise ValueError("No complete 30-sentence participant sheets were found.")

    human_array = np.asarray(baseline_human)
    baseline_array = np.asarray(baseline_predicted)
    final_array = np.asarray(final_predicted)
    return {
        "input": str(input_path),
        "model": str(model_path),
        "backend": backend or "none (baseline only)",
        "confidence_threshold": scoring.ARBITRATION_THRESHOLD,
        "complete_sheets": complete_sheets,
        "evaluated_sentences": len(human_array),
        "arbitrated_rows": arbitrated_rows,
        "baseline": metric_block(human_array, baseline_array),
        "with_arbitration": metric_block(human_array, final_array),
        "baseline_max_total_difference": max(row["baseline_total_difference"] for row in sheet_rows),
        "with_arbitration_max_total_difference": max(row["final_total_difference"] for row in sheet_rows),
        "sheet_results": sheet_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--backend", choices=["openai", "gemini"], default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for live OpenAI arbitration.")
    if args.backend == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required for live Gemini arbitration.")

    report = evaluate(args.input, args.model, args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sheet_results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())