"""test_sentence_agreement.py — GSoC Acceptance-Criteria Test Harness
=====================================================================

Validates the two primary GSoC KPIs for the Spanish EIT Automated Scoring Engine:

  KPI 1: ≥ 90% exact sentence-level agreement with experienced human raters
          (automated vs. human rubric application, measured per sentence).

  KPI 2: < 10-point difference in total EIT score per participant
          (on the 120-point scale = 30 sentences × 4 points).

The script evaluates every complete 30-sentence participant sheet in a workbook,
produces a rich terminal report with PASS/FAIL badges, and writes both a JSON
summary and a per-sheet CSV for archival / GSoC documentation.

Usage
-----
    # Baseline (no LLM) — recommended first run:
    python test_sentence_agreement.py

    # With LLM arbitration — Gemini backend (requires GOOGLE_API_KEY):
    python test_sentence_agreement.py --llm gemini

    # With LLM arbitration — OpenAI backend (requires OPENAI_API_KEY):
    python test_sentence_agreement.py --llm openai

    # Offline/CI mode — mock LLM (echoes the SVM score, no API key needed):
    python test_sentence_agreement.py --llm mock

    # Custom input workbook:
    python test_sentence_agreement.py --input data/my_reliability_data.xlsx

    # Adjust arbitration sensitivity (default: 0.70):
    python test_sentence_agreement.py --llm gemini --arbitration-threshold 0.65

Exit Codes
----------
    0 — All evaluated sheets pass both KPIs.
    1 — One or more sheets fail at least one KPI.
    2 — No compatible sheets were found in the input workbook.

Output Files
------------
    models/sentence_agreement_report.json          — full JSON summary
    models/sentence_agreement_sheet_results.csv    — per-sheet metrics table

Notes
-----
    - Only complete 30-sentence sheets (valid human scores for all 30 items)
      are included in the evaluation; partial sheets are logged and skipped.
    - Human score column auto-detection order: 'Final Rating' > 'Final Score'
      > 'Score Rater 1'.  Pass --human-column to override.
    - The 'Final_AutoScore' column (= LLM-arbitrated score when LLM is active,
      else the SVM score) is compared against the human rater.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

# Auto-load .env from the project root (silently skipped if python-dotenv not installed)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import score_responses as scoring


# ---------------------------------------------------------------------------
# Paths & thresholds
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "reliability_scored_test.xlsx"
DEFAULT_MODEL = ROOT / "models" / "ordinal_svm_pipeline.pkl"
DEFAULT_REPORT = ROOT / "models" / "sentence_agreement_report.json"
DEFAULT_SHEET_REPORT = ROOT / "models" / "sentence_agreement_sheet_results.csv"

# GSoC acceptance-criteria thresholds
KPI_AGREEMENT_THRESHOLD = 0.90   # ≥ 90% exact sentence-level agreement
KPI_TOTAL_DIFF_LIMIT = 10.0      # < 10-point total EIT score difference

EXPECTED_SENTENCES = 30          # sentences per complete participant sheet
TOTAL_SCORE_MAX = 120.0          # 30 sentences × 4 points

HUMAN_COLUMN_CANDIDATES = ("Final Rating", "Final Score", "Score Rater 1")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(path: Path):
    """Load a pickled OrdinalThresholdEnsemble, handling both module contexts."""
    main_module = sys.modules["__main__"]
    setattr(main_module, "OrdinalThresholdEnsemble", scoring.OrdinalThresholdEnsemble)
    with path.open("rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def detect_human_column(frame: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    """Return the first valid human-score column in the frame."""
    if requested:
        return requested if requested in frame.columns else None
    for candidate in HUMAN_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    return None


def has_transcription(frame: pd.DataFrame) -> bool:
    transcription_candidates = [
        "Final Transcription", "Transcription Rater 1", "Transcription",
    ]
    return any(c in frame.columns for c in transcription_candidates)


# ---------------------------------------------------------------------------
# Per-sheet evaluation
# ---------------------------------------------------------------------------


def evaluate_sheet(
    frame: pd.DataFrame,
    sheet_name: str,
    model,
    idea_units: pd.DataFrame,
    human_col: Optional[str],
    llm_backend: Optional[str],
    arbitration_threshold: float,
    max_delta: Optional[int],
) -> Optional[dict]:
    """Score one sheet and return a metrics dict, or None if not evaluable."""

    if human_col is None or "Stimulus" not in frame.columns or not has_transcription(frame):
        return None

    scored = scoring.score_dataframe(
        frame,
        model,
        idea_units,
        llm_backend=llm_backend,
        arbitration_threshold=arbitration_threshold,
        max_delta=max_delta,
    )

    human = pd.to_numeric(scored[human_col], errors="coerce")
    baseline = pd.to_numeric(scored["AutoScore"], errors="coerce")
    final = pd.to_numeric(scored["Final_AutoScore"], errors="coerce")
    valid = human.between(0, 4) & baseline.between(0, 4) & final.between(0, 4)

    n_valid = int(valid.sum())
    is_complete = (n_valid == EXPECTED_SENTENCES)

    if n_valid == 0:
        return None

    human_vals = human[valid].round().astype(int).to_numpy()
    baseline_vals = baseline[valid].round().astype(int).to_numpy()
    final_vals = final[valid].round().astype(int).to_numpy()

    def _metrics(predicted: np.ndarray) -> dict:
        return {
            "exact_agreement": float(accuracy_score(human_vals, predicted)),
            "within_one_agreement": float(np.mean(np.abs(human_vals - predicted) <= 1)),
            "qwk": float(cohen_kappa_score(human_vals, predicted, weights="quadratic")),
            "mae": float(mean_absolute_error(human_vals, predicted)),
            "human_total": int(human_vals.sum()),
            "auto_total": int(predicted.sum()),
            "total_difference": int(abs(predicted.sum() - human_vals.sum())),
        }

    baseline_metrics = _metrics(baseline_vals)
    final_metrics = _metrics(final_vals)
    n_arbitrated = int(scored.loc[valid, "Arbitrated"].sum()) if "Arbitrated" in scored.columns else 0

    # KPI assessment (only valid for complete 30-sentence sheets)
    agreement_pass = is_complete and final_metrics["exact_agreement"] >= KPI_AGREEMENT_THRESHOLD
    total_diff_pass = is_complete and final_metrics["total_difference"] < KPI_TOTAL_DIFF_LIMIT

    return {
        "sheet": sheet_name,
        "human_column": human_col,
        "valid_sentences": n_valid,
        "complete_30_sentence_sheet": is_complete,
        "n_arbitrated": n_arbitrated,
        # Baseline (SVM only)
        "baseline_exact_agreement": baseline_metrics["exact_agreement"],
        "baseline_within_one_agreement": baseline_metrics["within_one_agreement"],
        "baseline_qwk": baseline_metrics["qwk"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_human_total": baseline_metrics["human_total"],
        "baseline_auto_total": baseline_metrics["auto_total"],
        "baseline_total_difference": baseline_metrics["total_difference"],
        # Final (with arbitration if used)
        "final_exact_agreement": final_metrics["exact_agreement"],
        "final_within_one_agreement": final_metrics["within_one_agreement"],
        "final_qwk": final_metrics["qwk"],
        "final_mae": final_metrics["mae"],
        "final_human_total": final_metrics["human_total"],
        "final_auto_total": final_metrics["auto_total"],
        "final_total_difference": final_metrics["total_difference"],
        # KPI gates (only applicable for complete sheets)
        "kpi_agreement_pass": agreement_pass,
        "kpi_total_diff_pass": total_diff_pass,
        "kpi_overall_pass": agreement_pass and total_diff_pass,
    }


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------


def aggregate_results(sheet_rows: list[dict], llm_backend: Optional[str]) -> dict:
    """Compute aggregate metrics across all complete sheets."""
    complete = [r for r in sheet_rows if r["complete_30_sentence_sheet"]]
    all_rows = sheet_rows

    def _weighted_mean(key: str, rows: list[dict]) -> float:
        total_sentences = sum(r["valid_sentences"] for r in rows)
        if total_sentences == 0:
            return float("nan")
        return sum(r[key] * r["valid_sentences"] for r in rows) / total_sentences

    if not complete:
        return {
            "warning": "No complete 30-sentence sheets were found. KPI assessment requires complete sheets.",
            "all_sheets_evaluated": len(all_rows),
            "complete_sheets": 0,
        }

    return {
        # Coverage
        "total_sheets_evaluated": len(all_rows),
        "complete_sheets": len(complete),
        "total_sentences": sum(r["valid_sentences"] for r in complete),
        "total_arbitrated": sum(r["n_arbitrated"] for r in complete),
        # Baseline aggregate
        "baseline_micro_exact_agreement": _weighted_mean("baseline_exact_agreement", complete),
        "baseline_mean_exact_agreement": float(np.mean([r["baseline_exact_agreement"] for r in complete])),
        "baseline_min_exact_agreement": float(np.min([r["baseline_exact_agreement"] for r in complete])),
        "baseline_mean_qwk": float(np.mean([r["baseline_qwk"] for r in complete])),
        "baseline_mean_mae": float(np.mean([r["baseline_mae"] for r in complete])),
        "baseline_max_total_difference": int(max(r["baseline_total_difference"] for r in complete)),
        "baseline_mean_total_difference": float(np.mean([r["baseline_total_difference"] for r in complete])),
        # Final (with arbitration) aggregate
        "final_micro_exact_agreement": _weighted_mean("final_exact_agreement", complete),
        "final_mean_exact_agreement": float(np.mean([r["final_exact_agreement"] for r in complete])),
        "final_min_exact_agreement": float(np.min([r["final_exact_agreement"] for r in complete])),
        "final_mean_qwk": float(np.mean([r["final_qwk"] for r in complete])),
        "final_mean_mae": float(np.mean([r["final_mae"] for r in complete])),
        "final_max_total_difference": int(max(r["final_total_difference"] for r in complete)),
        "final_mean_total_difference": float(np.mean([r["final_total_difference"] for r in complete])),
        # KPI gate results
        "kpi_agreement_threshold": KPI_AGREEMENT_THRESHOLD,
        "kpi_total_diff_limit": KPI_TOTAL_DIFF_LIMIT,
        "kpi_agreement_pass": all(r["kpi_agreement_pass"] for r in complete),
        "kpi_total_diff_pass": all(r["kpi_total_diff_pass"] for r in complete),
        "kpi_overall_pass": all(r["kpi_overall_pass"] for r in complete),
        "sheets_failing_agreement": [r["sheet"] for r in complete if not r["kpi_agreement_pass"]],
        "sheets_failing_total_diff": [r["sheet"] for r in complete if not r["kpi_total_diff_pass"]],
        # Metadata
        "llm_backend": llm_backend or "none",
    }


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _badge(passing: bool) -> str:
    if passing:
        return f"{_GREEN}{_BOLD}PASS{_RESET}"
    return f"{_RED}{_BOLD}FAIL{_RESET}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _delta_label(baseline: float, final: float) -> str:
    diff = final - baseline
    if abs(diff) < 0.001:
        return "(no change)"
    sign = "+" if diff > 0 else ""
    return f"{_YELLOW}({sign}{diff * 100:.1f}pp with LLM){_RESET}"


def print_report(sheet_rows: list[dict], aggregate: dict, llm_backend: Optional[str]) -> None:
    """Print a human-readable terminal report."""
    complete = [r for r in sheet_rows if r["complete_30_sentence_sheet"]]
    skipped = [r for r in sheet_rows if not r["complete_30_sentence_sheet"]]
    has_llm = llm_backend is not None
    score_col = "final" if has_llm else "baseline"

    print()
    print("=" * 72)
    print(f"{_BOLD}  Spanish EIT — Sentence-Level Agreement Report{_RESET}")
    print(f"  LLM backend : {llm_backend or 'none (baseline SVM only)'}")
    print(f"  Evaluated   : {aggregate.get('complete_sheets', 0)} complete sheets "
          f"/ {aggregate.get('total_sheets_evaluated', 0)} total")
    if has_llm:
        print(f"  Arbitrated  : {aggregate.get('total_arbitrated', 0)} sentences "
              f"(confidence < {scoring.ARBITRATION_THRESHOLD:.2f})")
    print("=" * 72)

    # ── KPI Summary ────────────────────────────────────────────────────────
    print(f"\n{_BOLD}GSoC Acceptance Criteria{_RESET}")
    print(f"  {'KPI':<50} {'Target':<12} {'Result':<12} {'Status'}")
    print("  " + "-" * 68)

    min_agree = aggregate.get(f"final_min_exact_agreement" if has_llm else "baseline_min_exact_agreement", float("nan"))
    max_diff = aggregate.get("final_max_total_difference" if has_llm else "baseline_max_total_difference", float("nan"))
    agree_pass = aggregate.get("kpi_agreement_pass", False)
    diff_pass = aggregate.get("kpi_total_diff_pass", False)

    print(f"  {'KPI 1: Min exact sentence agreement (all sheets)':<50} "
          f"{'>=90.0%':<12} {_pct(min_agree):<12} {_badge(agree_pass)}")
    print(f"  {'KPI 2: Max total EIT score difference (per participant)':<50} "
          f"{'< 10 pts':<12} {max_diff} pts      {_badge(diff_pass)}")
    print()
    overall = agree_pass and diff_pass
    verdict = f"{_GREEN}{_BOLD}ALL KPIs PASSED ✓{_RESET}" if overall else f"{_RED}{_BOLD}ONE OR MORE KPIs FAILED ✗{_RESET}"
    print(f"  Overall result: {verdict}")

    # ── Per-Sheet Table ─────────────────────────────────────────────────────
    if complete:
        print(f"\n{_BOLD}Per-Sheet Results{_RESET}")
        header = f"  {'Sheet':<20} {'Agree':>8} {'W/in-1':>8} {'QWK':>7} {'MAE':>6} {'HumanΣ':>8} {'AutoΣ':>7} {'Δ':>5}  KPI"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in complete:
            agree = row[f"{score_col}_exact_agreement"]
            w1 = row[f"{score_col}_within_one_agreement"]
            qwk = row[f"{score_col}_qwk"]
            mae = row[f"{score_col}_mae"]
            h_total = row[f"{score_col}_human_total"]
            a_total = row[f"{score_col}_auto_total"]
            diff = row[f"{score_col}_total_difference"]
            kpi = _badge(row["kpi_overall_pass"])
            sheet_short = row["sheet"][:20]
            print(f"  {sheet_short:<20} {_pct(agree):>8} {_pct(w1):>8} {qwk:>7.3f} {mae:>6.3f} {h_total:>8} {a_total:>7} {diff:>5}  {kpi}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    if complete:
        print(f"\n{_BOLD}Aggregate (complete sheets){_RESET}")
        b = "baseline"
        f_ = "final"
        if has_llm:
            print(f"  Metric                     Baseline       With {llm_backend} arbitration")
            print("  " + "-" * 58)
            print(f"  Micro exact agreement    {_pct(aggregate[f'{b}_micro_exact_agreement']):>10}     "
                  f"{_pct(aggregate[f'{f_}_micro_exact_agreement']):>10}  "
                  f"{_delta_label(aggregate[f'{b}_micro_exact_agreement'], aggregate[f'{f_}_micro_exact_agreement'])}")
            print(f"  Min exact agreement      {_pct(aggregate[f'{b}_min_exact_agreement']):>10}     "
                  f"{_pct(aggregate[f'{f_}_min_exact_agreement']):>10}")
            print(f"  Mean QWK                 {aggregate[f'{b}_mean_qwk']:>10.4f}     "
                  f"{aggregate[f'{f_}_mean_qwk']:>10.4f}")
            print(f"  Mean MAE                 {aggregate[f'{b}_mean_mae']:>10.4f}     "
                  f"{aggregate[f'{f_}_mean_mae']:>10.4f}")
            print(f"  Max total difference     {aggregate[f'{b}_max_total_difference']:>10}     "
                  f"{aggregate[f'{f_}_max_total_difference']:>10}")
            print(f"  Mean total difference    {aggregate[f'{b}_mean_total_difference']:>10.2f}     "
                  f"{aggregate[f'{f_}_mean_total_difference']:>10.2f}")
        else:
            print(f"  Micro exact agreement : {_pct(aggregate[f'{b}_micro_exact_agreement'])}")
            print(f"  Min exact agreement   : {_pct(aggregate[f'{b}_min_exact_agreement'])}")
            print(f"  Mean QWK              : {aggregate[f'{b}_mean_qwk']:.4f}")
            print(f"  Mean MAE              : {aggregate[f'{b}_mean_mae']:.4f}")
            print(f"  Max total difference  : {aggregate[f'{b}_max_total_difference']} pts")
            print(f"  Mean total difference : {aggregate[f'{b}_mean_total_difference']:.2f} pts")

    # ── Skipped sheets ─────────────────────────────────────────────────────
    if skipped:
        print(f"\n{_YELLOW}Skipped sheets (incomplete or missing human scores):{_RESET}")
        for row in skipped:
            print(f"  {row['sheet']} — {row['valid_sentences']} valid sentences (need {EXPECTED_SENTENCES})")

    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    input_path: Path,
    model_path: Path,
    llm_backend: Optional[str],
    human_column: Optional[str],
    arbitration_threshold: float,
    max_delta: Optional[int],
) -> dict:
    """Run the full evaluation and return a report dict."""
    model = load_model(model_path)
    idea_units = scoring.load_idea_units(scoring.IDEA_UNITS_FILE)
    workbook = pd.ExcelFile(input_path)
    sheet_rows: list[dict] = []

    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name)
        human_col = detect_human_column(frame, human_column)
        result = evaluate_sheet(
            frame=frame,
            sheet_name=sheet_name,
            model=model,
            idea_units=idea_units,
            human_col=human_col,
            llm_backend=llm_backend,
            arbitration_threshold=arbitration_threshold,
            max_delta=max_delta,
        )
        if result is not None:
            sheet_rows.append(result)
        else:
            # Log incompatible sheets for transparency
            missing = []
            if "Stimulus" not in frame.columns:
                missing.append("Stimulus")
            if human_col is None:
                missing.append("human score column")
            if not has_transcription(frame):
                missing.append("transcription column")
            if missing:
                import logging
                logging.getLogger(__name__).info(
                    "Skipping sheet '%s' — missing: %s", sheet_name, ", ".join(missing)
                )

    if not sheet_rows:
        raise ValueError(
            "No compatible sheets found. Each sheet needs: Stimulus, a transcription "
            "column, and a human score column (Final Rating / Final Score / Score Rater 1)."
        )

    aggregate = aggregate_results(sheet_rows, llm_backend)

    return {
        "input": str(input_path),
        "model": str(model_path),
        "llm_backend": llm_backend or "none",
        "arbitration_threshold": arbitration_threshold,
        "arbitration_max_delta": max_delta,
        "kpi_criteria": {
            "min_exact_sentence_agreement": KPI_AGREEMENT_THRESHOLD,
            "max_total_score_difference_per_participant": KPI_TOTAL_DIFF_LIMIT,
            "total_score_scale": TOTAL_SCORE_MAX,
            "sentences_per_participant": EXPECTED_SENTENCES,
        },
        "aggregate": aggregate,
        "sheet_results": sheet_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input Excel workbook (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help=f"Pickled ordinal model (default: {DEFAULT_MODEL.name})",
    )
    parser.add_argument(
        "--llm",
        choices=["openai", "gemini", "mock"],
        default=None,
        help=(
            "Enable LLM arbitration for low-confidence sentences. "
            "'openai' requires OPENAI_API_KEY; 'gemini' requires GOOGLE_API_KEY; "
            "'mock' runs offline (echoes SVM score — no API key needed)."
        ),
    )
    parser.add_argument(
        "--arbitration-threshold",
        type=float,
        default=scoring.ARBITRATION_THRESHOLD,
        metavar="CONF",
        help=(
            f"Confidence threshold below which LLM arbitration is triggered "
            f"(default: {scoring.ARBITRATION_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--arbitration-max-delta",
        type=int,
        default=None,
        metavar="N",
        help="Maximum points the LLM score may differ from the SVM score. Default: no cap.",
    )
    parser.add_argument(
        "--human-column", default=None,
        help=(
            "Name of the human-score column to use (overrides auto-detection). "
            f"Auto-detects from: {', '.join(HUMAN_COLUMN_CANDIDATES)}."
        ),
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT,
        help=f"Output JSON report path (default: {DEFAULT_REPORT.name})",
    )
    parser.add_argument(
        "--sheet-report", type=Path, default=DEFAULT_SHEET_REPORT,
        help=f"Output CSV per-sheet results path (default: {DEFAULT_SHEET_REPORT.name})",
    )
    return parser.parse_args()


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Ensure stdout is UTF-8 on Windows (avoids cp1252 encode errors)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()

    # Guard API key requirements
    if args.llm == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is required for --llm openai.", file=sys.stderr)
        print("       Use --llm mock to test without an API key.", file=sys.stderr)
        return 2
    if args.llm == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable is required for --llm gemini.", file=sys.stderr)
        print("       Use --llm mock to test without an API key.", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
        print("       Run: python score_responses.py --retrain --input data/combined.xlsx --output /dev/null", file=sys.stderr)
        return 2

    try:
        report = evaluate(
            input_path=args.input,
            model_path=args.model,
            llm_backend=args.llm,
            human_column=args.human_column,
            arbitration_threshold=args.arbitration_threshold,
            max_delta=args.arbitration_max_delta,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Print terminal report
    print_report(report["sheet_results"], report["aggregate"], args.llm)

    # Write JSON report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"JSON report  → {args.report}")

    # Write per-sheet CSV
    sheet_df = pd.DataFrame(report["sheet_results"])
    sheet_df.to_csv(args.sheet_report, index=False)
    print(f"Sheet CSV    → {args.sheet_report}")
    print()

    return 0 if report["aggregate"].get("kpi_overall_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
