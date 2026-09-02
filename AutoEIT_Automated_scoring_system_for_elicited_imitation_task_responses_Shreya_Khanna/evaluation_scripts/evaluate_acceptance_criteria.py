"""Evaluate the Spanish EIT scorer against the project acceptance criteria.

The audit uses the saved ordinal model, scores every compatible sheet in a
reliability workbook, compares predictions with a human score column, and
checks both sentence-level agreement and participant-level EIT totals.

Example:
    python evaluate_acceptance_criteria.py
    python evaluate_acceptance_criteria.py --input data/reliability_scored_test.xlsx

The command writes a JSON summary and a per-sheet CSV under ``models/``. It
returns exit code 0 only when every evaluated sheet reaches at least 90% exact
sentence agreement and stays within 10 points of the human total.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

import score_responses as scoring


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "reliability_scored_test.xlsx"
DEFAULT_MODEL = ROOT / "models" / "ordinal_svm_pipeline.pkl"
DEFAULT_REPORT = ROOT / "models" / "acceptance_test_report.json"
DEFAULT_SHEET_REPORT = ROOT / "models" / "acceptance_test_sheet_results.csv"
AGREEMENT_THRESHOLD = 0.90
TOTAL_DIFFERENCE_LIMIT = 10.0
TOTAL_SCORE_MAX = 120.0
EXPECTED_SENTENCES = 30


def load_saved_model(path: Path):
    """Load a model pickled while ``score_responses.py`` ran as ``__main__``."""
    main_module = sys.modules["__main__"]
    setattr(main_module, "OrdinalThresholdEnsemble", scoring.OrdinalThresholdEnsemble)
    with path.open("rb") as handle:
        return pickle.load(handle)


def find_human_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        return requested if requested in frame.columns else None
    for candidate in ("Final Rating", "Final Score", "Score Rater 1"):
        if candidate in frame.columns:
            return candidate
    return None


def is_scored_sheet(frame: pd.DataFrame, human_column: str | None) -> bool:
    required = {"Stimulus"}
    return required.issubset(frame.columns) and human_column is not None


def evaluate_sheet(
    frame: pd.DataFrame,
    sheet_name: str,
    model,
    idea_units: pd.DataFrame,
    human_column: str | None,
) -> dict | None:
    if not is_scored_sheet(frame, human_column):
        return None

    result = scoring.score_dataframe(frame, model, idea_units)
    human = pd.to_numeric(result[human_column], errors="coerce")
    predicted = pd.to_numeric(result["Final_AutoScore"], errors="coerce")
    valid = human.between(0, 4) & predicted.between(0, 4)
    if not valid.any():
        return None

    human_values = human[valid].round().astype(int).to_numpy()
    predicted_values = predicted[valid].round().astype(int).to_numpy()
    total_human = float(human_values.sum())
    total_predicted = float(predicted_values.sum())
    total_difference = abs(total_predicted - total_human)
    exact_agreement = float(np.mean(predicted_values == human_values))
    within_one = float(np.mean(np.abs(predicted_values - human_values) <= 1))

    return {
        "sheet": sheet_name,
        "human_column": human_column,
        "valid_sentences": int(valid.sum()),
        "exact_sentence_agreement": exact_agreement,
        "within_one_sentence_agreement": within_one,
        "qwk": float(cohen_kappa_score(human_values, predicted_values, weights="quadratic")),
        "mae": float(mean_absolute_error(human_values, predicted_values)),
        "human_total": total_human,
        "automated_total": total_predicted,
        "total_difference": total_difference,
        "complete_120_point_eit": len(human_values) == EXPECTED_SENTENCES,
        "sentence_agreement_pass": exact_agreement >= AGREEMENT_THRESHOLD,
        "total_difference_pass": (
            len(human_values) == EXPECTED_SENTENCES
            and total_difference < TOTAL_DIFFERENCE_LIMIT
        ),
    }


def run_audit(input_path: Path, model_path: Path, human_column: str | None) -> dict:
    model = load_saved_model(model_path)
    idea_units = scoring.load_idea_units(scoring.IDEA_UNITS_FILE)
    workbook = pd.ExcelFile(input_path)
    rows: list[dict] = []

    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name)
        selected_human_column = find_human_column(frame, human_column)
        result = evaluate_sheet(frame, sheet_name, model, idea_units, selected_human_column)
        if result is not None:
            rows.append(result)

    if not rows:
        raise ValueError("No workbook sheets contained Stimulus and valid human score columns.")

    sheet_results = pd.DataFrame(rows)
    return {
        "input": str(input_path),
        "model": str(model_path),
        "model_class": type(model).__name__,
        "criteria": {
            "minimum_exact_sentence_agreement": AGREEMENT_THRESHOLD,
            "maximum_total_difference_exclusive": TOTAL_DIFFERENCE_LIMIT,
            "total_score_scale": TOTAL_SCORE_MAX,
            "sentences_per_120_point_eit": EXPECTED_SENTENCES,
        },
        "evaluated_sheets": int(len(sheet_results)),
        "evaluated_sentences": int(sheet_results["valid_sentences"].sum()),
        "exact_sentence_agreement_micro": float(
            sheet_results["exact_sentence_agreement"].mul(sheet_results["valid_sentences"]).sum()
            / sheet_results["valid_sentences"].sum()
        ),
        "mean_sheet_exact_sentence_agreement": float(sheet_results["exact_sentence_agreement"].mean()),
        "minimum_sheet_exact_sentence_agreement": float(sheet_results["exact_sentence_agreement"].min()),
        "maximum_total_difference": float(sheet_results["total_difference"].max()),
        "mean_total_difference": float(sheet_results["total_difference"].mean()),
        "sentence_agreement_pass": bool(sheet_results["sentence_agreement_pass"].all()),
        "total_difference_pass": bool(sheet_results["total_difference_pass"].all()),
        "acceptance_pass": bool(
            sheet_results["sentence_agreement_pass"].all()
            and sheet_results["total_difference_pass"].all()
        ),
        "sheet_results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--human-column", default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sheet-report", type=Path, default=DEFAULT_SHEET_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_audit(args.input, args.model, args.human_column)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["sheet_results"]).to_csv(args.sheet_report, index=False)
    print(json.dumps({key: value for key, value in report.items() if key != "sheet_results"}, indent=2))
    return 0 if report["acceptance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())