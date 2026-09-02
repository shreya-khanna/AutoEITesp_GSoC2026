"""Compare standalone text metrics with Ortega's human EIT scores.

This script deliberately does not train or load a predictive model. It treats
``combined.xlsx``'s ``final score`` as the human/Ortega score and evaluates
MER, METEOR, and Levenshtein similarity independently.

The output workbook is a copy of ``combined.xlsx`` with a new
``Metric_Agreement`` sheet containing Pearson/Spearman correlation, QWK, MAE,
and exact agreement after converting each metric to ordinal scores by its
training-data score-bin medians. The binning is descriptive, not a model.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
from jiwer import mer
from nltk.translate.meteor_score import meteor_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "combined.xlsx"
DEFAULT_OUTPUT = ROOT / "data" / "combined_metric_agreement.xlsx"


def clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_text(value: object) -> str:
    text = clean_text(value)
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9ñ\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein_similarity(reference: object, response: object) -> float:
    reference_text = clean_text(reference)
    response_text = clean_text(response)
    if reference_text is None or response_text is None:
        return np.nan
    distance = Levenshtein.distance(reference_text, response_text)
    return 1.0 - distance / max(len(reference_text), len(response_text), 1)


def meteor(reference: object, response: object) -> float:
    reference_text = normalize_text(reference)
    response_text = normalize_text(response)
    if not reference_text or not response_text:
        return np.nan
    try:
        return float(meteor_score([reference_text.split()], response_text.split()))
    except Exception:
        return np.nan


def metric_frame(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["ortega_score"] = pd.to_numeric(result["final score"], errors="coerce")
    result["mer"] = result.apply(
        lambda row: float(mer(normalize_text(row["stimulus"]), normalize_text(row["final transcription"])))
        if normalize_text(row["stimulus"]) and normalize_text(row["final transcription"])
        else np.nan,
        axis=1,
    )
    result["meteor"] = result.apply(
        lambda row: meteor(row["stimulus"], row["final transcription"]), axis=1
    )
    result["levenshtein_similarity"] = result.apply(
        lambda row: levenshtein_similarity(row["stimulus"], row["final transcription"]), axis=1
    )
    return result


def ordinal_from_metric(values: pd.Series, scores: pd.Series) -> tuple[pd.Series, list[float]]:
    """Map a similarity/error metric to 0-4 using score-bin medians.

    This is a transparent descriptive conversion used only to report exact
    ordinal agreement; it does not fit a classifier or tune thresholds.
    """
    valid = values.notna() & scores.between(0, 4)
    medians = scores[valid].groupby(scores[valid]).apply(lambda group: values[group.index].median())
    ordered = medians.sort_values().to_numpy()
    if len(ordered) != 5:
        raise ValueError("Expected all five Ortega score classes for ordinal comparison.")
    distances = np.abs(values.to_numpy()[:, None] - ordered[None, :])
    predictions = np.argmin(distances, axis=1)
    predictions[values.isna().to_numpy()] = -1
    return pd.Series(predictions, index=values.index), ordered.tolist()


def evaluate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = metric_frame(data)
    rows = []
    for metric in ["mer", "meteor", "levenshtein_similarity"]:
        valid = enriched[[metric, "ortega_score"]].dropna()
        valid = valid[valid["ortega_score"].between(0, 4)]
        x = valid[metric]
        y = valid["ortega_score"].astype(int)
        higher_is_better = metric != "mer"
        ordinal_values = enriched[metric] if higher_is_better else -enriched[metric]
        predictions, score_bin_medians = ordinal_from_metric(ordinal_values, enriched["ortega_score"])
        ordinal_valid = (
            (predictions >= 0)
            & enriched["ortega_score"].between(0, 4)
            & enriched[metric].notna()
        )
        predicted = predictions[ordinal_valid].astype(int)
        human = enriched.loc[ordinal_valid, "ortega_score"].astype(int)
        rows.append({
            "metric": metric,
            "direction_for_agreement": "higher is better" if higher_is_better else "lower is better",
            "n": len(valid),
            "pearson_r": pearsonr(x, y).statistic,
            "pearson_p": pearsonr(x, y).pvalue,
            "spearman_r": spearmanr(x, y).statistic,
            "spearman_p": spearmanr(x, y).pvalue,
            "qwk_after_ordinal_bin_medians": cohen_kappa_score(human, predicted, weights="quadratic"),
            "mae_after_ordinal_bin_medians": mean_absolute_error(human, predicted),
            "exact_agreement_after_ordinal_bin_medians": float(np.mean(human.to_numpy() == predicted.to_numpy())),
            "score_bin_medians_low_to_high": ", ".join(f"{value:.6f}" for value in score_bin_medians),
        })
    return enriched, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = pd.read_excel(args.input)
    required = {"stimulus", "final transcription", "final score"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    enriched, summary = evaluate(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Original_Data", index=False)
        enriched.to_excel(writer, sheet_name="Metric_Values", index=False)
        summary.to_excel(writer, sheet_name="Metric_Agreement", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()