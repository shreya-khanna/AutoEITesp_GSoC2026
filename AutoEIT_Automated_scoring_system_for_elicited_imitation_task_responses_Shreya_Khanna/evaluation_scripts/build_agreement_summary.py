"""Build a sentence-level Ortega agreement summary for all local variants.

Measures are evaluated directly against ``combined.xlsx``'s ``final score``.
Classical models are evaluated with five stimulus-grouped folds, so a response
is scored only by a model that did not train on that stimulus. Historical
embedding results are copied from ``models/summary_table.csv`` and labelled
as historical because their embeddings are not recomputed by this script.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
from jiwer import mer, wer
from nltk.translate.meteor_score import meteor_score
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import score_responses as scoring


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "combined.xlsx"
OUTPUT = ROOT / "data" / "ortega_agreement_summary.xlsx"


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", clean(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9ñ\s]", " ", text).strip()


def tokens(value: object) -> list[str]:
    return normalized(value).split()


def metric_values(data: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    reference_tokens = data["stimulus"].map(tokens)
    response_tokens = data["final transcription"].map(tokens)

    def safe_metric(function, reference, response):
        if not reference or not response:
            return np.nan
        try:
            return float(function(reference, response))
        except Exception:
            return np.nan

            reference_text = data["stimulus"].map(normalized)
            response_text = data["final transcription"].map(normalized)
            result["WER"] = [safe_metric(wer, r, h) for r, h in zip(reference_text, response_text)]
            result["MER"] = [safe_metric(mer, r, h) for r, h in zip(reference_text, response_text)]
    result["Levenshtein similarity"] = [
        1 - Levenshtein.distance(clean(r), clean(h)) / max(len(clean(r)), len(clean(h)), 1)
        for r, h in zip(data["stimulus"], data["final transcription"])
    ]
    result["LCS ratio"] = [
        SequenceMatcher(None, r, h).find_longest_match(0, len(r), 0, len(h)).size / max(len(r), 1)
        for r, h in zip(reference_tokens, response_tokens)
    ]
    result["Word Jaccard"] = [
        len(set(r) & set(h)) / max(len(set(r) | set(h)), 1)
        for r, h in zip(reference_tokens, response_tokens)
    ]
    result["METEOR"] = [
        safe_metric(lambda r, h: meteor_score([r], h), r, h)
        for r, h in zip(reference_tokens, response_tokens)
    ]
    result["Token-sort similarity"] = [
        SequenceMatcher(None, " ".join(sorted(r)), " ".join(sorted(h))).ratio()
        for r, h in zip(reference_tokens, response_tokens)
    ]
    result["Syllable ratio"] = data.apply(
        lambda row: scoring.count_spanish_syllables(row["final transcription"])
        / max(scoring.extract_stimulus_syllables(row["stimulus"]), 1), axis=1
    )
    result["Response length ratio"] = [len(h) / max(len(r), 1) for r, h in zip(reference_tokens, response_tokens)]
    result["Character similarity"] = [
        SequenceMatcher(None, clean(r).lower(), clean(h).lower()).ratio()
        for r, h in zip(data["stimulus"], data["final transcription"])
    ]
    stemmer = SnowballStemmer("spanish")
    result["CER"] = [
        Levenshtein.distance(clean(r), clean(h)) / max(len(clean(r)), 1)
        for r, h in zip(data["stimulus"], data["final transcription"])
    ]
    result["Stemmed overlap"] = [
        len({stemmer.stem(word) for word in r} & {stemmer.stem(word) for word in h})
        / max(len({stemmer.stem(word) for word in r}), 1)
        for r, h in zip(reference_tokens, response_tokens)
    ]
    disfluencies = {"xxx", "pause", "gibberish", "inaudible", "eh", "uh", "um", "este", "mmm"}
    result["Disfluency count"] = [len(set(h) & disfluencies) for h in response_tokens]
    result["Bigram overlap"] = [
        len(set(zip(r, r[1:])) & set(zip(h, h[1:]))) / max(len(set(zip(r, r[1:]))), 1)
        for r, h in zip(reference_tokens, response_tokens)
    ]
    result["Negation change"] = [
        int((set(r) & scoring.NEGATION_WORDS) != (set(h) & scoring.NEGATION_WORDS))
        for r, h in zip(reference_tokens, response_tokens)
    ]
    result["Negation preserved"] = [
        int(not (set(r) & scoring.NEGATION_WORDS) or bool((set(r) & scoring.NEGATION_WORDS) & set(h)))
        for r, h in zip(reference_tokens, response_tokens)
    ]
    result["Plural preserved"] = [
        int(not ({scoring.normalize_token(t) for t in r if scoring.looks_plural(t)}
                 - {scoring.normalize_token(t) for t in h}))
        for r, h in zip(reference_tokens, response_tokens)
    ]
    return result


def measure_agreement(values: pd.Series, scores: pd.Series, name: str) -> dict:
    valid = values.notna() & scores.between(0, 4)
    x = values[valid].astype(float)
    y = scores[valid].astype(int)
    # Error measures are reversed so higher transformed values mean better performance.
    oriented = -x if name in {"WER", "MER", "CER", "Disfluency count", "Negation change"} else x
    medians = y.groupby(y).apply(lambda group: oriented[group.index].median()).sort_values()
    predicted = np.argmin(np.abs(oriented.to_numpy()[:, None] - medians.to_numpy()[None, :]), axis=1)
    predicted = pd.Series(predicted, index=oriented.index)
    predicted = predicted[valid]
    return {
        "category": "measure",
        "variant": name,
        "evaluation": "all valid combined.xlsx rows",
        "n": int(valid.sum()),
        "exact_agreement": float(np.mean(predicted.to_numpy() == y.to_numpy())),
        "qwk": float(cohen_kappa_score(y, predicted, weights="quadratic")),
        "mae": float(mean_absolute_error(y, predicted)),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_r": float(spearmanr(x, y).statistic),
    }


def model_agreement(data: pd.DataFrame, feature_table: pd.DataFrame) -> list[dict]:
    y = data["final score"].astype(int).to_numpy()
    groups = data["stimulus_key"].to_numpy()
    models = {
        "Ordinal threshold ensemble (11 features)": scoring.OrdinalThresholdEnsemble(
            thresholds=scoring.THRESHOLDS, C=0.5, random_state=scoring.RANDOM_STATE
        ),
        "Logistic regression (11 features)": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=scoring.RANDOM_STATE
        ),
        "LinearSVC (11 features)": LinearSVC(class_weight="balanced", random_state=scoring.RANDOM_STATE),
    }
    results = []
    for name, estimator in models.items():
        predictions = np.full(len(data), -1, dtype=int)
        for train_idx, test_idx in GroupKFold(n_splits=5).split(feature_table, y, groups):
            estimator.fit(feature_table.iloc[train_idx], y[train_idx])
            predictions[test_idx] = estimator.predict(feature_table.iloc[test_idx])
        results.append({
            "category": "classical model",
            "variant": name,
            "evaluation": "5-fold stimulus-grouped CV",
            "n": len(y),
            "exact_agreement": float(np.mean(predictions == y)),
            "qwk": float(cohen_kappa_score(y, predictions, weights="quadratic")),
            "mae": float(mean_absolute_error(y, predictions)),
            "pearson_r": float(pearsonr(predictions, y).statistic),
            "spearman_r": float(spearmanr(predictions, y).statistic),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    data = scoring.load_and_prepare_data(args.input, scoring.IDEA_UNITS_FILE)
    measures = metric_values(data)
    measure_rows = [measure_agreement(measures[name], data["final score"], name) for name in measures]
    features = data.apply(scoring.build_feature_row, axis=1)[scoring.FEATURE_COLUMNS].astype(float)
    model_rows = model_agreement(data, features)
    summary = pd.DataFrame(measure_rows + model_rows).sort_values("qwk", ascending=False)

    historical = pd.read_csv(ROOT / "models" / "summary_table.csv")
    historical.insert(0, "category", "historical model result")
    historical["evaluation"] = "published repository aggregate; not re-executed"
    historical = historical.rename(columns={"model": "variant", "quadratic_kappa": "qwk"})
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Sentence_Agreement", index=False)
        historical.to_excel(writer, sheet_name="Historical_Results", index=False)
        measures.assign(ortega_score=data["final score"].to_numpy()).to_excel(
            writer, sheet_name="Measure_Values", index=False
        )
    print(summary.to_string(index=False))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()