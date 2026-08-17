"""
Render docs/asr_model_comparison_ranking.csv (in turn a rendering of §1.1 of
docs/asr_model_comparison_analysis.md) as a standalone table image, for sharing the ranking
outside the full write-up. Previously a one-off, hand-built image (see the "Add a ranking-table
image and CSV" commit); rebuilt as a script here so it can be regenerated whenever the CSV changes
instead of drifting out of sync with it silently.

Highlights the best row(s) (lowest mean_wer, ties included) in blue and the worst row in red/pink,
matching the original hand-built image's convention.

Usage:
    python -m src.postprocessing.build_asr_ranking_table_image
"""
import csv
import logging

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.utils.paths import data_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CSV_PATH = "docs/asr_model_comparison_ranking.csv"
OUT_PATH = "docs/plots/asr_model_comparison_ranking_table.png"

HEADER_COLOR = "#1a3a6b"
ROW_COLORS = ["#f7f7f5", "#ffffff"]
BEST_COLOR = "#e3edf9"
WORST_COLOR = "#fbe7e7"
BEST_TEXT_COLOR = "#1a5fb4"
WORST_TEXT_COLOR = "#c01c28"

COLUMNS = [
    ("rank", "Rank", 0.0, 0.045, "center"),
    ("model", "Model", 0.045, 0.235, "left"),
    ("mean_wer", "Mean\nWER", 0.235, 0.305, "center"),
    ("mean_mer", "Mean\nMER", 0.305, 0.375, "center"),
    ("mean_cer", "Mean\nCER", 0.375, 0.445, "center"),
    ("median_wer", "Median\nWER", 0.445, 0.515, "center"),
    ("pct_word_perfect", "% Word-\nPerfect", 0.515, 0.585, "center"),
    ("decoder_family", "Decoder family", 0.585, 0.735, "left"),
    ("training_language_breadth", "Training language breadth", 0.735, 0.935, "left"),
    ("eval_scope", "Eval scope", 0.935, 1.0, "left"),
]


def load_rows() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()
    n_rows = len(rows)

    model_wers = [float(r["mean_wer"]) for r in rows if r["rank"]]
    best_wer = min(model_wers)
    worst_wer = max(model_wers)

    row_h = 0.9
    header_h = 1.3
    title_h = 1.6
    footer_h = 0.6
    total_h = title_h + header_h + n_rows * row_h + footer_h

    fig_w = 20.0
    fig_h = total_h * (fig_w / 20.0) * 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.axis("off")

    ax.text(0.0, title_h * 0.35, "ASR Model Comparison — Full Ranking Table",
             fontsize=24, fontweight="bold", va="center", ha="left")
    ax.text(0.0, title_h * 0.75,
             "Resegmented data, 102 hit recordings, mean/median over both human raters "
             "(docs/asr_model_comparison_analysis.md §1.1)",
             fontsize=13, color="#555555", va="center", ha="left")

    y = title_h
    ax.add_patch(Rectangle((0, y), 1, header_h, facecolor=HEADER_COLOR, edgecolor="none"))
    for key, label, x0, x1, align in COLUMNS:
        tx = x0 + 0.008 if align == "left" else (x0 + x1) / 2
        ax.text(tx, y + header_h / 2, label, fontsize=13, fontweight="bold", color="white",
                 va="center", ha=align)
    y += header_h

    for i, row in enumerate(rows):
        is_human = not row["rank"]
        wer = float(row["mean_wer"])
        is_best = (not is_human) and wer == best_wer
        is_worst = (not is_human) and wer == worst_wer

        if is_best:
            bg = BEST_COLOR
        elif is_worst:
            bg = WORST_COLOR
        else:
            bg = ROW_COLORS[i % 2]
        ax.add_patch(Rectangle((0, y), 1, row_h, facecolor=bg, edgecolor="#dddddd", linewidth=0.5))

        wer_color = BEST_TEXT_COLOR if is_best else (WORST_TEXT_COLOR if is_worst else "#111111")

        for key, _, x0, x1, align in COLUMNS:
            val = row[key]
            tx = x0 + 0.008 if align == "left" else (x0 + x1) / 2
            if key == "rank":
                ax.text(tx, y + row_h / 2, val if val else "—", fontsize=13, va="center", ha=align,
                         color="#555555")
            elif key == "model":
                ax.text(tx, y + row_h * 0.32, row["model"], fontsize=13, fontweight="bold",
                         va="center", ha=align, color="#111111")
                ax.text(tx, y + row_h * 0.72, row["checkpoint"], fontsize=10.5, va="center",
                         ha=align, color="#777777")
            elif key == "mean_wer":
                ax.text(tx, y + row_h / 2, val, fontsize=13, fontweight="bold", va="center",
                         ha=align, color=wer_color)
            elif key == "decoder_family":
                ax.text(tx, y + row_h / 2, val, fontsize=10, va="center", ha=align,
                         color="#333333")
            elif key == "training_language_breadth":
                ax.text(tx, y + row_h / 2, val, fontsize=9, va="center", ha=align,
                         color="#333333")
            elif key == "eval_scope":
                ax.text(tx, y + row_h / 2, val, fontsize=10.5, va="center", ha=align,
                         color="#111111")
            else:
                ax.text(tx, y + row_h / 2, val, fontsize=13, va="center", ha=align, color="#111111")
        y += row_h

    ax.text(0.0, y + footer_h * 0.6,
             "WER / MER / CER: lower is better  |  % Word-Perfect: higher is better  |  "
             "“full” = all 184 valid resegmented recordings transcribed; "
             "“hit-only” = the 102 scoreable recordings only",
             fontsize=11, color="#555555", va="center", ha="left")

    fig.tight_layout(pad=1.2)
    fig.savefig(OUT_PATH, dpi=140, facecolor="white")
    log.info("Wrote %d row(s) to %s", n_rows, OUT_PATH)


if __name__ == "__main__":
    main()
