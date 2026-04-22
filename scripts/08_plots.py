"""
Script 08: Generate all report figures.

Figures produced:
  Fig 1 — Grouped bar: Macro-F1 per model per variant
  Fig 2 — Delta heatmaps (3 models side-by-side)
  Fig 3 — Per-class F1 radar: clean vs. combined (worst) per model
  Fig 4 — Identity mention bias: mean P(hate) by group × model
  Fig 5 — CDA confidence delta distribution (boxplot)
  Fig 6 — AAVE vs. SAE False Positive Rate (bar)
  Fig 7 — Register recall: clean vs. register_shift per model
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from math import pi

os.makedirs("report/figures", exist_ok=True)

# ── Styling ──────────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "bert_hatexplain":   "BERT-HateXplain",
    "twitter_roberta":   "Twitter-RoBERTa",
    "dynabench_roberta": "DynaBench-RoBERTa",
}
VARIANT_LABELS = {
    "clean":          "Clean",
    "typos":          "Typos",
    "paraphrase":     "Paraphrase",
    "word_order":     "Word Order",
    "combined":       "Combined",
    "register_shift": "Register Shift",
    "code_mixed":     "Code-Mixed",
    "demo_swap":      "Demo. Swap",
}

PALETTE  = ["#4C72B0", "#DD8452", "#55A868"]
sns.set_theme(style="whitegrid", font_scale=1.1)

MODELS   = ["bert_hatexplain", "twitter_roberta", "dynabench_roberta"]
VARIANTS = list(VARIANT_LABELS.keys())

def load_summary():
    path = "scores/aggregate/summary_table.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_delta():
    path = "scores/aggregate/delta_table.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Fig 1: Grouped bar — Macro-F1 per model per variant ──────────────────────
def fig1_macro_f1(df):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(VARIANTS))
    w = 0.26
    offsets = [-w, 0, w]

    for i, model in enumerate(MODELS):
        vals = []
        for v in VARIANTS:
            row = df[(df["model"] == model) & (df["variant"] == v)]
            vals.append(float(row["macro_f1"].iloc[0]) if not row.empty else 0.0)
        bars = ax.bar(x + offsets[i], vals, w - 0.02, label=MODEL_LABELS[model],
                      color=PALETTE[i], edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=25, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fig 1 — Macro-F1 Across Dataset Variants and Models")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="upper right")
    plt.tight_layout()
    plt.savefig("report/figures/fig1_macro_f1.pdf", dpi=200)
    plt.savefig("report/figures/fig1_macro_f1.png", dpi=200)
    plt.close()
    print("Saved Fig 1 — Macro-F1 bar chart")


# ── Fig 2: Δ-Heatmaps (3 models) ─────────────────────────────────────────────
def fig2_delta_heatmaps(df_delta):
    metrics = ["delta_accuracy", "delta_macro_f1", "delta_hate_f1",
               "delta_offensive_f1", "delta_normal_f1"]
    metric_labels = ["ΔAcc", "ΔMacro-F1", "ΔHate-F1", "ΔOffensive-F1", "ΔNormal-F1"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, model in zip(axes, MODELS):
        sub = df_delta[df_delta["model"] == model]
        matrix = []
        for v in VARIANTS:
            row = sub[sub["variant"] == v]
            if row.empty:
                matrix.append([0.0] * len(metrics))
            else:
                matrix.append([float(row[m].iloc[0]) if row[m].iloc[0] is not None else 0.0
                                for m in metrics])
        mat = np.array(matrix)
        vmax = max(0.05, np.abs(mat).max())
        sns.heatmap(mat, ax=ax, cmap="RdYlGn", center=0,
                    vmin=-vmax, vmax=vmax,
                    xticklabels=metric_labels,
                    yticklabels=[VARIANT_LABELS[v] for v in VARIANTS],
                    annot=True, fmt=".3f", annot_kws={"size": 8},
                    linewidths=0.4, linecolor="grey", cbar=True)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Fig 2 — Δ-Score Heatmaps (Perturbed − Clean)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("report/figures/fig2_delta_heatmaps.pdf", dpi=200, bbox_inches="tight")
    plt.savefig("report/figures/fig2_delta_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved Fig 2 — delta heatmaps")


# ── Fig 3: Radar chart — per-class F1 clean vs. combined ─────────────────────
def fig3_radar(df):
    categories = ["Hate-F1", "Offensive-F1", "Normal-F1"]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                              subplot_kw=dict(polar=True))

    for ax, model in zip(axes, MODELS):
        for variant, color, ls in [("clean", "#4C72B0", "-"), ("combined", "#C44E52", "--")]:
            row = df[(df["model"] == model) & (df["variant"] == variant)]
            if row.empty:
                continue
            vals = [float(row["hate_f1"].iloc[0]),
                    float(row["offensive_f1"].iloc[0]),
                    float(row["normal_f1"].iloc[0])]
            vals += [vals[0]]
            ax.plot(angles, vals, color=color, linestyle=ls, linewidth=2,
                    label=VARIANT_LABELS[variant])
            ax.fill(angles, vals, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7)
        ax.set_title(MODEL_LABELS[model], size=10, pad=12, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    fig.suptitle("Fig 3 — Per-Class F1: Clean vs. Combined Perturbation", y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig("report/figures/fig3_radar.pdf", dpi=200, bbox_inches="tight")
    plt.savefig("report/figures/fig3_radar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved Fig 3 — radar chart")


# ── Fig 4: Identity mention bias ──────────────────────────────────────────────
def fig4_identity_bias():
    path = "scores/fairness/identity_bias_summary.csv"
    if not os.path.exists(path):
        print("  [skip] Fig 4 — identity_bias_summary.csv not found")
        return
    data = pd.read_csv(path)
    groups = sorted(data["group"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = data[data["model"] == model].set_index("group")
        means = [sub.loc[g, "mean_hate_prob"] if g in sub.index else 0 for g in groups]
        stds  = [sub.loc[g, "std_hate_prob"]  if g in sub.index else 0 for g in groups]
        y_pos = np.arange(len(groups))
        bars = ax.barh(y_pos, means, xerr=stds, color=PALETTE[MODELS.index(model)],
                       edgecolor="white", linewidth=0.6, capsize=3, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(0.5, color="red", linewidth=0.8, linestyle="--",
                   label="Decision threshold (0.5)")
        ax.set_xlabel("Mean P(hate_speech)")
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        if MODELS.index(model) == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Fig 4 — Identity Mention Bias: P(hate) for Neutral Sentences", fontsize=12)
    plt.tight_layout()
    plt.savefig("report/figures/fig4_identity_bias.pdf", dpi=200, bbox_inches="tight")
    plt.savefig("report/figures/fig4_identity_bias.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved Fig 4 — identity bias")


# ── Fig 5: CDA confidence delta boxplot ───────────────────────────────────────
def fig5_cda_boxplot():
    frames = []
    for model in MODELS:
        path = f"scores/fairness/cda_detail_{model}.csv"
        if os.path.exists(path):
            d = pd.read_csv(path)
            d["model"] = MODEL_LABELS[model]
            frames.append(d)
    if not frames:
        print("  [skip] Fig 5 — no CDA detail files found")
        return
    combined = pd.concat(frames)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=combined, x="abs_delta", y="model", palette=PALETTE,
                orient="h", ax=ax, width=0.5, flierprops={"marker": ".", "markersize": 3})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|Δ P(hate_speech)| (original − swapped)")
    ax.set_ylabel("")
    ax.set_title("Fig 5 — Counterfactual Demographic Swap: Confidence Asymmetry")
    plt.tight_layout()
    plt.savefig("report/figures/fig5_cda_boxplot.pdf", dpi=200)
    plt.savefig("report/figures/fig5_cda_boxplot.png", dpi=200)
    plt.close()
    print("Saved Fig 5 — CDA boxplot")


# ── Fig 6: AAVE vs. SAE FPR ───────────────────────────────────────────────────
def fig6_aave_fpr():
    path = "scores/fairness/aave_fpr.csv"
    if not os.path.exists(path):
        print("  [skip] Fig 6 — aave_fpr.csv not found")
        return
    data = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(data))
    w = 0.35
    bars1 = ax.bar(x - w/2, data["aave_fpr"], w, label="AAVE", color="#C44E52", edgecolor="white")
    bars2 = ax.bar(x + w/2, data["sae_fpr"],  w, label="SAE",  color="#4C72B0", edgecolor="white")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in data["model"]], rotation=10)
    ax.set_ylabel("False Positive Rate")
    ax.set_ylim(0, min(1.0, data[["aave_fpr","sae_fpr"]].max().max() + 0.15))
    ax.set_title("Fig 6 — AAVE vs. SAE False Positive Rate on Normal-Class Posts")
    ax.legend()
    plt.tight_layout()
    plt.savefig("report/figures/fig6_aave_fpr.pdf", dpi=200)
    plt.savefig("report/figures/fig6_aave_fpr.png", dpi=200)
    plt.close()
    print("Saved Fig 6 — AAVE FPR")


# ── Fig 7: Register shift recall ──────────────────────────────────────────────
def fig7_register_recall():
    path = "scores/fairness/register_recall.csv"
    if not os.path.exists(path):
        print("  [skip] Fig 7 — register_recall.csv not found")
        return
    data = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(data))
    w = 0.35
    ax.bar(x - w/2, data["clean_recall"],    w, label="Clean",          color="#4C72B0", edgecolor="white")
    ax.bar(x + w/2, data["register_recall"], w, label="Register Shift", color="#DD8452", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in data["model"]], rotation=10)
    ax.set_ylabel("Hate Speech Recall")
    ax.set_ylim(0, 1.1)
    ax.set_title("Fig 7 — Hate Speech Recall: Clean vs. Register-Shifted Input")
    ax.legend()
    plt.tight_layout()
    plt.savefig("report/figures/fig7_register_recall.pdf", dpi=200)
    plt.savefig("report/figures/fig7_register_recall.png", dpi=200)
    plt.close()
    print("Saved Fig 7 — register recall")


def main():
    df_summary = load_summary()
    df_delta   = load_delta()

    if df_summary is not None:
        fig1_macro_f1(df_summary)
        fig3_radar(df_summary)
    else:
        print("[skip] Figs 1, 3 — summary_table.csv not found")

    if df_delta is not None:
        fig2_delta_heatmaps(df_delta)
    else:
        print("[skip] Fig 2 — delta_table.csv not found")

    fig4_identity_bias()
    fig5_cda_boxplot()
    fig6_aave_fpr()
    fig7_register_recall()

    print("\nAll figures saved to report/figures/")

if __name__ == "__main__":
    main()
