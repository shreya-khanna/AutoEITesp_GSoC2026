"""
score_responses.py — Automated Spanish EIT Scoring Pipeline
============================================================

Usage
-----
    python score_responses.py --input path/to/responses.xlsx --output scored_output.xlsx

    # With LLM arbitration — Gemini (set GOOGLE_API_KEY):
    python score_responses.py --input responses.xlsx --output scored.xlsx --llm gemini

    # With LLM arbitration — OpenAI (set OPENAI_API_KEY):
    python score_responses.py --input responses.xlsx --output scored.xlsx --llm openai

    # Dry-run / offline mock (no API key needed — LLM echoes the SVM score):
    python score_responses.py --input responses.xlsx --output scored.xlsx --llm mock

    # Tune arbitration sensitivity (default: 0.70):
    python score_responses.py --input responses.xlsx --output scored.xlsx \
        --llm gemini --arbitration-threshold 0.65

    # Cap how far the LLM can move the score from the SVM (default: no cap):
    python score_responses.py --input responses.xlsx --output scored.xlsx \
        --llm gemini --arbitration-max-delta 1

    # Score a specific sheet in a multi-sheet workbook:
    python score_responses.py --input data.xlsx --sheet "030.016-1A" --output scored.xlsx

    # Re-train model from scratch (requires combined.xlsx and idea_units_Spanish_AutoEIT.xlsx in data/):
    python score_responses.py --retrain --output scored.xlsx --input data/combined.xlsx

Input format
------------
    The input Excel file must have at minimum two columns (case-insensitive):
        - 'Stimulus'           : the EIT stimulus sentence
        - 'Final Transcription': the learner's spoken response (transcription)

    Optionally, if the file contains 'Score Rater 1' and 'Score Rater 2', those are
    preserved in the output for comparison.

Output format
-------------
    An Excel file with all original columns preserved, plus:
        - 'AutoScore'         : predicted score (0-4) from the ordinal SVM
        - 'LLM_Score'         : LLM-arbitrated score (only if --llm flag is used)
        - 'Final_AutoScore'   : the chosen final score (LLM score if arbitrated, else AutoScore)
        - 'Confidence'        : model confidence (max probability from threshold models)
        - 'Arbitrated'        : True/False — whether LLM was invoked for this row

    When --llm is used, a sidecar transparency CSV is also written next to the output
    file (suffix _llm_arbitration_log.csv) containing per-row evidence:
        row_index, stimulus, transcription, svm_score, confidence, llm_score, delta

Model
-----
    The pipeline uses an OrdinalThresholdEnsemble of four binary logistic regressions,
    one for each threshold P(score >= k) for k in {1, 2, 3, 4}. The final score is
    the sum of thresholds exceeded.

    Features (11 total, all computed from stimulus + transcription text):
        coverage_fraction, coverage_count, wer, substitutions, insertions,
        deletions, hit_rate, syllable_ratio, response_length_ratio,
        negation_preserved, plural_preserved

    Cross-validated performance (stimulus-grouped split, n=1964 responses):
        Train QWK: 0.8957  |  Val QWK: 0.8906  |  Test QWK: 0.8329

LLM Arbitration
---------------
    When --llm is specified, the LLM is invoked only for responses where the
    model confidence is below the ARBITRATION_THRESHOLD (default: 0.70). This
    targets the hardest cases (typically scores near decision boundaries 1<->2, 2<->3)
    while keeping API costs low.

    Supported backends:
        --llm openai   : uses gpt-4o-mini (set OPENAI_API_KEY)
        --llm gemini   : uses gemini-1.5-flash (set GOOGLE_API_KEY)
                         Works with both google-generativeai (v0) and google-genai (v1) SDKs.
        --llm mock     : echoes the SVM score; no API calls (useful for CI / offline testing)

    Retries: each backend retries up to 3 times with exponential backoff on transient errors.
    If all retries fail, the SVM score is kept and Arbitrated is set to False.

    Delta cap: --arbitration-max-delta N restricts LLM adjustments to at most N points
    away from the SVM score. Scores outside the cap fall back to the SVM score.

Reproducibility
---------------
    Model is retrained with random_state=42. The same seed is used for the
    stimulus-grouped GroupShuffleSplit so train/val/test splits are identical
    to the notebook experiments.

    To pin the model to exact notebook results, use --retrain to rebuild from
    data/combined.xlsx with the same feature engineering code.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Auto-load .env from the project root (silently skipped if python-dotenv not installed)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "ordinal_svm_pipeline.pkl"
DATA_DIR = Path(__file__).parent / "data"
IDEA_UNITS_FILE = Path(__file__).parent.parent / "data" / "idea_units_Spanish_AutoEIT.xlsx"
COMBINED_FILE = DATA_DIR / "combined.xlsx"

FEATURE_COLUMNS = [
    "coverage_fraction",
    "coverage_count",
    "wer",
    "substitutions",
    "insertions",
    "deletions",
    "hit_rate",
    "syllable_ratio",
    "response_length_ratio",
    "negation_preserved",
    "plural_preserved",
]

THRESHOLDS = [1, 2, 3, 4]
RANDOM_STATE = 42
ARBITRATION_THRESHOLD = 0.70  # invoke LLM when confidence < this
ARBITRATION_MAX_RETRIES = 3    # retry attempts per API call
ARBITRATION_MAX_DELTA = None   # None = no cap; set to int to restrict score movement

NEGATION_WORDS = {"no", "nunca", "jamas", "nadie", "ningun", "ninguna", "ninguno", "sin", "ni"}

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zñ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize_for_match(text).split()


def normalize_token(token: str) -> str:
    token = token.strip()
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def looks_plural(token: str) -> bool:
    token = normalize_token(token)
    return len(token) > 3 and (token.endswith("s") or token.endswith("es"))


def lemmatize_tokens(text: str) -> list[str]:
    return [normalize_token(token) for token in tokenize(text)]


def count_spanish_syllables(text: str) -> int:
    cleaned = normalize_for_match(text)
    if not cleaned:
        return 0
    try:
        import silabeador
        total = 0
        for word in cleaned.split():
            if word:  # guard against empty strings
                try:
                    total += len(silabeador.syllabify(word))
                except Exception:
                    # fallback for problematic tokens
                    groups = re.findall(r"[aeiou]+", word)
                    total += max(1, len(groups))
        return total
    except ImportError:
        total = 0
        for word in cleaned.split():
            groups = re.findall(r"[aeiou]+", word)
            total += max(1, len(groups))
        return total


def extract_stimulus_syllables(stimulus_text: str) -> int:
    match = re.search(r"\((\d+)\)\s*$", str(stimulus_text))
    if match:
        return int(match.group(1))
    return count_spanish_syllables(stimulus_text)


# ---------------------------------------------------------------------------
# Alignment (custom WER)
# ---------------------------------------------------------------------------


def compute_word_alignment(reference_tokens: list, hypothesis_tokens: list) -> tuple:
    rows, cols = len(reference_tokens), len(hypothesis_tokens)
    costs = np.zeros((rows + 1, cols + 1), dtype=int)
    backtrack = np.zeros((rows + 1, cols + 1), dtype=np.int8)

    for row in range(1, rows + 1):
        costs[row, 0] = row
        backtrack[row, 0] = 1
    for col in range(1, cols + 1):
        costs[0, col] = col
        backtrack[0, col] = 2

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            if reference_tokens[row - 1] == hypothesis_tokens[col - 1]:
                sub_cost = costs[row - 1, col - 1]
            else:
                sub_cost = costs[row - 1, col - 1] + 1
            del_cost = costs[row - 1, col] + 1
            ins_cost = costs[row, col - 1] + 1
            best = min(sub_cost, del_cost, ins_cost)
            costs[row, col] = best
            if best == sub_cost:
                backtrack[row, col] = 0 if reference_tokens[row - 1] == hypothesis_tokens[col - 1] else 3
            elif best == del_cost:
                backtrack[row, col] = 1
            else:
                backtrack[row, col] = 2

    hits = substitutions = insertions = deletions = 0
    row, col = rows, cols
    while row > 0 or col > 0:
        step = backtrack[row, col]
        if step == 0:
            hits += 1; row -= 1; col -= 1
        elif step == 3:
            substitutions += 1; row -= 1; col -= 1
        elif step == 1:
            deletions += 1; row -= 1
        else:
            insertions += 1; col -= 1

    total_reference = max(len(reference_tokens), 1)
    wer = (substitutions + insertions + deletions) / total_reference
    hit_rate = hits / total_reference
    return hits, substitutions, insertions, deletions, wer, hit_rate


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------


def calculate_coverage_metrics(transcription: str, idea_units_breakdown) -> tuple:
    idea_units = []
    if pd.notna(idea_units_breakdown):
        cleaned = re.sub(r"\(\d+\)\s*$", "", str(idea_units_breakdown).strip())
        idea_units = [unit.strip() for unit in cleaned.split("::") if unit.strip()]

    if not idea_units:
        return 0.0, 0, []

    response_tokens = set(lemmatize_tokens(transcription))
    covered_units = []
    for unit in idea_units:
        unit_tokens = [t for t in lemmatize_tokens(unit) if t]
        if unit_tokens and all(t in response_tokens for t in unit_tokens):
            covered_units.append(unit)

    coverage_count = len(covered_units)
    coverage_fraction = coverage_count / len(idea_units)
    return coverage_fraction, coverage_count, covered_units


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_feature_row(row: pd.Series) -> pd.Series:
    stimulus = str(row["stimulus"])
    response = str(row["final transcription"])

    stimulus_tokens = lemmatize_tokens(stimulus)
    response_tokens = lemmatize_tokens(response)

    hits, subs, ins, dels, wer, hit_rate = compute_word_alignment(stimulus_tokens, response_tokens)
    coverage_fraction, coverage_count, _ = calculate_coverage_metrics(
        response, row.get("idea_units_breakdown", None)
    )

    stimulus_syllables = extract_stimulus_syllables(stimulus)
    response_syllables = count_spanish_syllables(response)
    response_length_ratio = len(response_tokens) / max(len(stimulus_tokens), 1)
    syllable_ratio = response_syllables / max(stimulus_syllables, 1)

    prompt_negations = {t for t in stimulus_tokens if t in NEGATION_WORDS}
    response_negations = {t for t in response_tokens if t in NEGATION_WORDS}
    negation_preserved = int(not prompt_negations or bool(prompt_negations & response_negations))

    prompt_plural_tokens = {normalize_token(t) for t in stimulus_tokens if looks_plural(t)}
    response_plural_tokens = {normalize_token(t) for t in response_tokens}
    plural_preserved = int(not prompt_plural_tokens or prompt_plural_tokens.issubset(response_plural_tokens))

    return pd.Series({
        "coverage_fraction": coverage_fraction,
        "coverage_count": coverage_count,
        "wer": wer,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "hit_rate": hit_rate,
        "syllable_ratio": syllable_ratio,
        "response_length_ratio": response_length_ratio,
        "negation_preserved": negation_preserved,
        "plural_preserved": plural_preserved,
    })


# ---------------------------------------------------------------------------
# Ordinal model
# ---------------------------------------------------------------------------


class OrdinalThresholdEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ordinal classifier using K-1 binary logistic regressions.

    For each threshold k in {1, 2, 3, 4}, trains P(score >= k).
    Prediction: sum of thresholds exceeded (via probability >= 0.5).
    Monotonicity is enforced via cumulative minimum on probabilities.
    """

    def __init__(self, thresholds: list = None, C: float = 0.5, random_state: int = 42):
        self.thresholds = thresholds or [1, 2, 3, 4]
        self.C = C
        self.random_state = random_state
        self.models: dict = {}

    def fit(self, X, y):
        self.models = {}
        for threshold in self.thresholds:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=self.C,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=self.random_state,
                )),
            ])
            model.fit(X, (y >= threshold).astype(int))
            self.models[threshold] = model
        return self

    def predict_threshold_probabilities(self, X) -> np.ndarray:
        cols = [self.models[t].predict_proba(X)[:, 1] for t in self.thresholds]
        probs = np.column_stack(cols)
        # enforce monotonicity: P(>=k) should be non-increasing in k
        probs = np.minimum.accumulate(probs, axis=1)
        return probs

    def predict_confidence(self, X) -> np.ndarray:
        """Return a normalized confidence proxy in the range [0, 1]."""
        probs = self.predict_threshold_probabilities(X)
        # Distance from 0.5 is at most 0.5; scale it so the arbitration
        # threshold has the intended [0, 1] interpretation.
        distances = np.abs(probs - 0.5)
        return np.clip(distances.mean(axis=1) * 2.0, 0.0, 1.0)

    def predict(self, X) -> np.ndarray:
        return (self.predict_threshold_probabilities(X) >= 0.5).sum(axis=1)


# ---------------------------------------------------------------------------
# Model training and loading
# ---------------------------------------------------------------------------


def load_idea_units(idea_units_path: Path) -> pd.DataFrame:
    df = pd.read_excel(idea_units_path)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {c: "idea_units_breakdown" for c in df.columns if "idea" in c and "unit" in c}
    if col_map:
        df = df.rename(columns=col_map)
    df["stimulus_key"] = df["stimulus"].apply(normalize_for_match)
    return df[["stimulus_key", "idea_units_breakdown"]].drop_duplicates("stimulus_key")


def load_and_prepare_data(data_path: Path, idea_units_path: Path) -> pd.DataFrame:
    df = pd.read_excel(data_path)
    df.columns = df.columns.str.strip().str.lower()
    df["final score"] = pd.to_numeric(df["final score"], errors="coerce")
    df = df.dropna(subset=["stimulus", "final transcription", "final score"]).copy()
    df["final score"] = df["final score"].astype(int)
    df = df[df["final score"].between(0, 4)]

    idea_units_df = load_idea_units(idea_units_path)
    df["stimulus_key"] = df["stimulus"].apply(normalize_for_match)
    df = df.merge(idea_units_df, on="stimulus_key", how="left")
    return df


def train_model(df: pd.DataFrame) -> OrdinalThresholdEnsemble:
    logger.info("Engineering features for %d responses…", len(df))
    features = df.apply(build_feature_row, axis=1)
    df = pd.concat([df, features], axis=1)

    X = df[FEATURE_COLUMNS].astype(float).values
    y = df["final score"].values

    model = OrdinalThresholdEnsemble(thresholds=THRESHOLDS, C=0.5, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def save_model(model: OrdinalThresholdEnsemble, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> OrdinalThresholdEnsemble:
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model


def get_or_train_model(retrain: bool = False) -> OrdinalThresholdEnsemble:
    if not retrain and MODEL_PATH.exists():
        try:
            return load_model(MODEL_PATH)
        except Exception as e:
            logger.warning("Failed to load saved model (%s). Retraining…", e)

    logger.info("Training ordinal SVM model from %s…", COMBINED_FILE)
    if not COMBINED_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found at {COMBINED_FILE}. "
            "Please provide combined.xlsx in the data/ directory."
        )
    if not IDEA_UNITS_FILE.exists():
        raise FileNotFoundError(
            f"Idea units file not found at {IDEA_UNITS_FILE}. "
            "Please provide idea_units_Spanish_AutoEIT.xlsx in the data/ directory."
        )

    df = load_and_prepare_data(COMBINED_FILE, IDEA_UNITS_FILE)
    model = train_model(df)

    # Evaluate on held-out test split
    _evaluate_model_on_test(model, df)

    save_model(model, MODEL_PATH)
    return model


def _evaluate_model_on_test(model: OrdinalThresholdEnsemble, df: pd.DataFrame) -> None:
    features = df.apply(build_feature_row, axis=1)
    df = pd.concat([df.copy(), features], axis=1)
    X = df[FEATURE_COLUMNS].astype(float).values
    y = df["final score"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    _, test_idx = next(splitter.split(df, groups=df["stimulus_key"]))
    X_test = X[test_idx]
    y_test = y[test_idx]
    test_pred = model.predict(X_test)

    qwk = cohen_kappa_score(y_test, test_pred, weights="quadratic")
    mae = mean_absolute_error(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred)
    logger.info("Test metrics — Accuracy: %.4f | MAE: %.4f | QWK: %.4f", acc, mae, qwk)


# ---------------------------------------------------------------------------
# LLM Arbitration
# ---------------------------------------------------------------------------


def _build_scoring_prompt(
    stimulus: str,
    transcription: str,
    svm_score: int,
    features: Optional[dict] = None,
) -> str:
    """Build the LLM arbitration prompt with Ortega (2015) rubric and optional feature evidence."""
    feature_block = ""
    if features:
        feature_block = f"""
QUANTITATIVE EVIDENCE (computed by the scoring engine):
  Word error rate (WER)         : {features.get('wer', 'N/A'):.3f}  (0=perfect, 1=fully wrong)
  Hit rate (correct words)      : {features.get('hit_rate', 'N/A'):.3f}  (1=all words reproduced)
  Coverage fraction (idea units): {features.get('coverage_fraction', 'N/A'):.3f}  (1=all ideas present)
  Syllable ratio                : {features.get('syllable_ratio', 'N/A'):.3f}  (1=same length)
  Negation preserved            : {features.get('negation_preserved', 'N/A')}  (1=yes, 0=changed)
  Plural preserved              : {features.get('plural_preserved', 'N/A')}  (1=yes, 0=changed)
"""

    return f"""You are an expert scorer for a Spanish Elicited Imitation Task (EIT).
You are resolving an ambiguous automated score. Apply the rubric carefully.

SCORING RUBRIC (Ortega 2015, 0-4 scale):
  4 = Perfect or near-perfect reproduction.
        All content words (nouns, verbs, adjectives, adverbs) are present and accurate.
        Morphology (gender, number, tense, person) matches the stimulus.
        Syntactic structure is fully preserved.
  3 = Minor deviation only.
        At most one content word missing, substituted, or morphologically incorrect.
        Overall meaning and structure are intact.
  2 = Partial reproduction.
        Multiple content words missing or wrong; OR significant morphological errors;
        OR major syntactic restructuring that alters meaning.
  1 = Minimal content preserved.
        Very few correct content words; response is fragmentary or heavily restructured.
  0 = No meaningful content.
        Response is unintelligible, off-topic, completely empty, or a non-Spanish utterance.

SCORING DECISION GUIDE:
  - Score on CONTENT accuracy (content words, morphology, syntax) — NOT pronunciation.
  - Filler words ("um", "este", pauses) are ignored.
  - A student who reproduces the meaning with minor wording differences scores higher
    than one who uses the exact words in the wrong order.
  - When in doubt between two adjacent scores, choose the lower score.

STIMULUS (what the student heard):
{stimulus}

STUDENT RESPONSE (what they said):
{transcription}
{feature_block}
The automated model assigned a preliminary score of {svm_score}.
This case was flagged as low-confidence. Review the response against the rubric and provide your score.

Respond with ONLY a single integer (0, 1, 2, 3, or 4). No explanation."""


def _call_openai(prompt: str, max_retries: int = ARBITRATION_MAX_RETRIES) -> Optional[int]:
    """Call OpenAI gpt-4o-mini with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r"[0-4]", content)
            if match is None:
                raise ValueError(f"No valid score in response: {content!r}")
            return int(match.group())
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("OpenAI attempt %d/%d failed (%s). Retrying in %ds…", attempt + 1, max_retries, e, wait)
                time.sleep(wait)
            else:
                logger.warning("OpenAI arbitration failed after %d attempts: %s", max_retries, e)
    return None


def _call_gemini_v0(prompt: str) -> Optional[int]:
    """Gemini using the legacy google-generativeai (v0) SDK."""
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-flash-latest")
    response = model.generate_content(prompt)
    content = response.text.strip()
    match = re.search(r"[0-4]", content)
    if match is None:
        raise ValueError(f"No valid score in response: {content!r}")
    return int(match.group())


def _call_gemini_v1(prompt: str) -> Optional[int]:
    """Gemini using the new google-genai (v1) SDK."""
    from google import genai  # type: ignore
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )
    content = response.text.strip()
    match = re.search(r"[0-4]", content)
    if match is None:
        raise ValueError(f"No valid score in response: {content!r}")
    return int(match.group())


def _call_gemini(prompt: str, max_retries: int = ARBITRATION_MAX_RETRIES) -> Optional[int]:
    """Call Gemini with exponential backoff retry. Auto-detects v0 vs v1 SDK."""
    for attempt in range(max_retries):
        try:
            # Prevent hitting the 15 RPM free-tier rate limit
            time.sleep(4)
            # Prefer v1 SDK (google-genai), fall back to v0 (google-generativeai)
            try:
                return _call_gemini_v1(prompt)
            except ImportError:
                return _call_gemini_v0(prompt)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Gemini attempt %d/%d failed (%s). Retrying in %ds…", attempt + 1, max_retries, e, wait)
                time.sleep(wait)
            else:
                logger.warning("Gemini arbitration failed after %d attempts: %s", max_retries, e)
    return None


def _call_mock(svm_score: int) -> int:
    """Mock backend: echo the SVM score (no API call). Used for offline testing / CI."""
    return svm_score


def arbitrate_with_llm(
    stimulus: str,
    transcription: str,
    svm_score: int,
    confidence: float,
    backend: str,
    features: Optional[dict] = None,
    arbitration_threshold: float = ARBITRATION_THRESHOLD,
    max_delta: Optional[int] = ARBITRATION_MAX_DELTA,
) -> tuple[Optional[int], bool]:
    """
    Returns (llm_score, was_arbitrated).

    Only calls the LLM when confidence < arbitration_threshold.
    If max_delta is set, the LLM score is clamped so it cannot move
    more than max_delta points from the SVM score; out-of-range scores
    fall back to the SVM score (was_arbitrated=False).

    Supported backends: 'openai', 'gemini', 'mock'.
    """
    if confidence >= arbitration_threshold:
        return None, False

    if backend == "mock":
        return _call_mock(svm_score), True

    prompt = _build_scoring_prompt(stimulus, transcription, svm_score, features=features)

    if backend == "openai":
        llm_score = _call_openai(prompt)
    elif backend == "gemini":
        llm_score = _call_gemini(prompt)
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. Use 'openai', 'gemini', or 'mock'.")

    if llm_score is None:
        return None, False

    # Apply delta cap
    if max_delta is not None and abs(llm_score - svm_score) > max_delta:
        logger.debug(
            "LLM score %d exceeds max_delta=%d from SVM score %d — keeping SVM score.",
            llm_score, max_delta, svm_score,
        )
        return None, False

    return llm_score, True


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _normalise_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def load_input_excel(path: Path, sheet: Optional[str] = None) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Load input Excel. Returns (df, detected_sheet_name, all_sheet_names).
    Normalizes column names to lowercase for internal processing.
    """
    xl = pd.ExcelFile(path)
    sheet_names = xl.sheet_names

    if sheet is not None:
        if sheet not in sheet_names:
            raise ValueError(f"Sheet '{sheet}' not found. Available: {sheet_names}")
        target_sheet = sheet
    else:
        # Pick the first non-empty sheet
        target_sheet = None
        for s in sheet_names:
            df_tmp = xl.parse(s)
            if not df_tmp.empty and len(df_tmp.columns) > 1:
                target_sheet = s
                break
        if target_sheet is None:
            raise ValueError("No non-empty sheets found in the input file.")

    df = xl.parse(target_sheet)
    return df, target_sheet, sheet_names


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    col_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------


def score_dataframe(
    df: pd.DataFrame,
    model: OrdinalThresholdEnsemble,
    idea_units_df: Optional[pd.DataFrame],
    llm_backend: Optional[str] = None,
    arbitration_threshold: Optional[float] = None,
    max_delta: Optional[int] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame with Stimulus and Final Transcription columns,
    add AutoScore, Confidence, and optionally LLM_Score / Final_AutoScore.
    """
    stimulus_col = _find_column(df, ["Stimulus", "stimulus"])
    transcription_col = _find_column(df, [
        "Final Transcription", "final transcription", "Final_Transcription",
        "Transcription Rater 1", "transcription rater 1",  # 038 reliability sheets
        "Transcription",
    ])

    if stimulus_col is None or transcription_col is None:
        missing = []
        if stimulus_col is None:
            missing.append("'Stimulus'")
        if transcription_col is None:
            missing.append("'Final Transcription' (or 'Transcription Rater 1')")
        raise ValueError(
            f"Input file is missing required columns: {', '.join(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Build working copy with standardized names
    work_df = df.copy()
    work_df["stimulus"] = work_df[stimulus_col].astype(str)
    work_df["final transcription"] = work_df[transcription_col].astype(str)

    # Join idea units if available
    if idea_units_df is not None:
        work_df["stimulus_key"] = work_df["stimulus"].apply(normalize_for_match)
        work_df = work_df.merge(idea_units_df, on="stimulus_key", how="left")
    else:
        work_df["idea_units_breakdown"] = None

    # Feature engineering
    logger.info("Computing features for %d rows…", len(work_df))
    features = work_df.apply(build_feature_row, axis=1)
    X = features[FEATURE_COLUMNS].astype(float).values

    # Predict
    svm_scores = model.predict(X)
    confidences = model.predict_confidence(X)

    # Assign initial scores
    work_df["AutoScore"] = svm_scores
    work_df["Confidence"] = confidences.round(4)
    work_df["Arbitrated"] = False

    if llm_backend:
        _arb_threshold = arbitration_threshold if arbitration_threshold is not None else ARBITRATION_THRESHOLD
        _max_delta = max_delta
        logger.info(
            "Running LLM arbitration (%s) for rows with confidence < %.2f…",
            llm_backend, _arb_threshold,
        )
        llm_scores = []
        arbitrated_flags = []
        transparency_log: list[dict] = []
        low_conf_count = 0

        # Pre-compute features dict per row for transparency + prompt enrichment
        feature_records = features[FEATURE_COLUMNS].to_dict(orient="records")

        for i, (_, row_data) in enumerate(work_df.iterrows()):
            svm_score = int(svm_scores[i])
            confidence = float(confidences[i])
            stimulus = row_data["stimulus"]
            transcription = row_data["final transcription"]
            feat_dict = feature_records[i]

            llm_score, was_arbitrated = arbitrate_with_llm(
                stimulus=stimulus,
                transcription=transcription,
                svm_score=svm_score,
                confidence=confidence,
                backend=llm_backend,
                features=feat_dict,
                arbitration_threshold=_arb_threshold,
                max_delta=_max_delta,
            )
            llm_scores.append(llm_score)
            arbitrated_flags.append(was_arbitrated)
            if was_arbitrated:
                low_conf_count += 1

            transparency_log.append({
                "row_index": i,
                "stimulus": str(stimulus)[:80],
                "transcription": str(transcription)[:80],
                "svm_score": svm_score,
                "confidence": round(confidence, 4),
                "arbitrated": was_arbitrated,
                "llm_score": llm_score,
                "delta": (llm_score - svm_score) if was_arbitrated and llm_score is not None else None,
            })

        logger.info("LLM arbitrated %d / %d rows.", low_conf_count, len(work_df))
        work_df["LLM_Score"] = llm_scores
        work_df["Arbitrated"] = arbitrated_flags
        work_df["Final_AutoScore"] = work_df.apply(
            lambda r: int(r["LLM_Score"]) if r["Arbitrated"] and r["LLM_Score"] is not None else int(r["AutoScore"]),
            axis=1,
        )
        # Attach transparency log for caller to persist
        work_df.attrs["_llm_transparency_log"] = transparency_log
    else:
        work_df["Final_AutoScore"] = work_df["AutoScore"]

    # Copy scored columns back to original df (preserving all original columns)
    result_df = df.copy()
    score_cols = ["AutoScore", "Confidence", "Arbitrated", "Final_AutoScore"]
    if llm_backend:
        score_cols.insert(2, "LLM_Score")

    for col in score_cols:
        result_df[col] = work_df[col].values

    return result_df


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_output(
    original_path: Path,
    result_df: pd.DataFrame,
    sheet_name: str,
    all_sheet_names: list[str],
    output_path: Path,
    model_info: dict,
) -> None:
    """
    Write output Excel preserving all original sheets, modifying only the target sheet.
    Adds a 'Model Info' sheet summarising the run.
    """
    xl = pd.ExcelFile(original_path)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Write all original sheets, replacing the target one with scored version
        for s in all_sheet_names:
            if s == sheet_name:
                result_df.to_excel(writer, sheet_name=s, index=False)
            else:
                other_df = xl.parse(s)
                if not other_df.empty:
                    other_df.to_excel(writer, sheet_name=s, index=False)

        # Add model info sheet
        info_df = pd.DataFrame([
            {"Parameter": "Model", "Value": "OrdinalThresholdEnsemble (4 binary logistic regressions)"},
            {"Parameter": "Features", "Value": ", ".join(FEATURE_COLUMNS)},
            {"Parameter": "Thresholds", "Value": str(THRESHOLDS)},
            {"Parameter": "C (regularization)", "Value": "0.5"},
            {"Parameter": "Test QWK (canonical split)", "Value": "0.8329"},
            {"Parameter": "Test MAE (canonical split)", "Value": "0.4902"},
            {"Parameter": "Training data", "Value": str(COMBINED_FILE)},
            {"Parameter": "LLM backend", "Value": model_info.get("llm_backend", "None")},
            {"Parameter": "Arbitration threshold", "Value": str(ARBITRATION_THRESHOLD)},
            {"Parameter": "Scored sheet", "Value": sheet_name},
            {"Parameter": "Total rows scored", "Value": str(model_info.get("n_rows", "?"))},
        ])
        info_df.to_excel(writer, sheet_name="AutoScore_Info", index=False)

    logger.info("Output written to %s", output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Spanish EIT scorer (Ordinal SVM + optional LLM arbitration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input Excel file path")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output Excel file path")
    parser.add_argument("--sheet", "-s", default=None, help="Sheet name to score (default: first non-empty sheet)")
    parser.add_argument(
        "--llm",
        choices=["openai", "gemini", "mock"],
        default=None,
        help=(
            "Enable LLM arbitration. "
            "'openai' requires OPENAI_API_KEY; 'gemini' requires GOOGLE_API_KEY; "
            "'mock' runs offline (echoes SVM score — useful for CI/testing)."
        ),
    )
    parser.add_argument(
        "--arbitration-threshold",
        type=float,
        default=ARBITRATION_THRESHOLD,
        metavar="CONF",
        help=(
            f"Confidence threshold below which LLM arbitration is triggered "
            f"(default: {ARBITRATION_THRESHOLD}). Lower values = fewer LLM calls."
        ),
    )
    parser.add_argument(
        "--arbitration-max-delta",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of points the LLM score may differ from the SVM score. "
            "Scores outside this range fall back to the SVM score. Default: no cap."
        ),
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-training the model from data/combined.xlsx even if a saved model exists.",
    )
    parser.add_argument(
        "--no-idea-units",
        action="store_true",
        help="Skip joining idea units (faster, but coverage_fraction and coverage_count will be 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load or train model
    model = get_or_train_model(retrain=args.retrain)

    # Load idea units
    idea_units_df = None
    if not args.no_idea_units:
        if IDEA_UNITS_FILE.exists():
            idea_units_df = load_idea_units(IDEA_UNITS_FILE)
            logger.info("Loaded %d idea-unit entries.", len(idea_units_df))
        else:
            logger.warning(
                "Idea units file not found at %s. Coverage features will be 0. "
                "Use --no-idea-units to suppress this warning.",
                IDEA_UNITS_FILE,
            )

    # Load input
    logger.info("Loading input: %s", input_path)
    df, sheet_name, all_sheet_names = load_input_excel(input_path, sheet=args.sheet)
    logger.info("Scoring sheet '%s' (%d rows)…", sheet_name, len(df))

    # Score
    result_df = score_dataframe(
        df,
        model,
        idea_units_df,
        llm_backend=args.llm,
        arbitration_threshold=args.arbitration_threshold,
        max_delta=args.arbitration_max_delta,
    )

    # Print summary
    scores = result_df["Final_AutoScore"]
    logger.info("\n--- Scoring Summary ---")
    logger.info("Total responses scored: %d", len(scores))
    logger.info("Score distribution:\n%s", scores.value_counts().sort_index().to_string())
    if args.llm:
        n_arbitrated = result_df["Arbitrated"].sum()
        logger.info("LLM-arbitrated: %d / %d (%.1f%%)", n_arbitrated, len(scores), 100 * n_arbitrated / len(scores))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(
        original_path=input_path,
        result_df=result_df,
        sheet_name=sheet_name,
        all_sheet_names=all_sheet_names,
        output_path=output_path,
        model_info={
            "llm_backend": args.llm or "None",
            "n_rows": len(df),
            "arbitration_threshold": args.arbitration_threshold,
            "arbitration_max_delta": args.arbitration_max_delta,
        },
    )

    # Write per-row transparency CSV sidecar when LLM arbitration was used
    if args.llm:
        transparency_log = result_df.attrs.get("_llm_transparency_log", [])
        if transparency_log:
            log_path = output_path.with_name(output_path.stem + "_llm_arbitration_log.csv")
            with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["row_index", "stimulus", "transcription", "svm_score",
                              "confidence", "arbitrated", "llm_score", "delta"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(transparency_log)
            logger.info("LLM arbitration transparency log written to %s", log_path)

    print(f"\nDone! Scored output saved to: {output_path}")
    if args.llm:
        n_arbitrated = result_df["Arbitrated"].sum()
        print(f"  LLM arbitrated {n_arbitrated} uncertain cases.")
        if args.llm != "mock":
            log_path = output_path.with_name(output_path.stem + "_llm_arbitration_log.csv")
            print(f"  Transparency log: {log_path}")


if __name__ == "__main__":
    main()
