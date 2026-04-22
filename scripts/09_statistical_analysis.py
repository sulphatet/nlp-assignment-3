"""
Script 09: Statistical significance analysis and supplementary tables.

Computes:
  - McNemar's test for each model × variant (clean vs. perturbed)
  - Effect size (Cohen's g / odds ratio from McNemar table)
  - Failure-case counts per model × variant × type
  - Confidence intervals for Macro-F1 using bootstrap (n=1000)
  - Per-label degradation breakdown

Outputs:
  scores/aggregate/mcnemar_results.csv
  scores/aggregate/failure_case_counts.csv
  scores/aggregate/bootstrap_ci.csv
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.utils import resample

MODELS   = ["bert_hatexplain", "twitter_roberta", "dynabench_roberta"]
VARIANTS = ["typos", "paraphrase", "word_order", "combined",
            "register_shift", "code_mixed", "demo_swap"]
LABELS   = ["hate_speech", "offensive", "normal"]

os.makedirs("scores/aggregate", exist_ok=True)

def load_outputs(path):
    if not os.path.exists(path):
        return None
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


# ── McNemar's test ────────────────────────────────────────────────────────────
def mcnemar_test(clean_recs, pert_recs):
    """
    Returns b (degradation), c (recovery), p-value (two-sided exact binomial).
    b = correct_clean AND wrong_pert
    c = wrong_clean AND correct_pert
    """
    clean_by_id = {r["id"]: r for r in clean_recs}
    pert_by_id  = {r["id"]: r for r in pert_recs}
    b = c_val = 0
    for rid, orig in clean_by_id.items():
        p = pert_by_id.get(rid)
        if p is None:
            continue
        clean_ok = orig["pred_label"] == orig["gold_label"]
        pert_ok  = p["pred_label"]    == p["gold_label"]
        if clean_ok and not pert_ok:
            b += 1
        elif not clean_ok and pert_ok:
            c_val += 1
    n = b + c_val
    if n < 5:
        pval = 1.0
    else:
        pval = min(1.0, 2.0 * float(binom.cdf(min(b, c_val), n, 0.5)))
    # Odds ratio from McNemar (b/c); guard div-by-zero
    odds_ratio = round(b / c_val, 3) if c_val > 0 else None
    return {"b": b, "c": c_val, "n_discordant": n,
            "p_value": round(pval, 5), "odds_ratio": odds_ratio,
            "significant_alpha_05": pval < 0.05}


# ── Bootstrap CI for Macro-F1 ─────────────────────────────────────────────────
def bootstrap_macro_f1(records, n_bootstrap=1000, ci=0.95):
    """Return (mean, lower, upper) bootstrap CI for Macro-F1."""
    from sklearn.metrics import f1_score
    golds = [r["gold_label"] for r in records]
    preds = [r["pred_label"] for r in records]
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(golds), size=len(golds), replace=True)
        g_s = [golds[i] for i in idx]
        p_s = [preds[i] for i in idx]
        scores.append(f1_score(g_s, p_s, average="macro",
                                labels=LABELS, zero_division=0))
    alpha = 1 - ci
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return round(float(np.mean(scores)), 4), round(lo, 4), round(hi, 4)


# ── Per-label degradation breakdown ──────────────────────────────────────────
def label_degradation(clean_recs, pert_recs):
    """
    For each gold label, count how many correct→wrong examples there are.
    """
    clean_by_id = {r["id"]: r for r in clean_recs}
    pert_by_id  = {r["id"]: r for r in pert_recs}
    counts = {lbl: 0 for lbl in LABELS}
    totals = {lbl: 0 for lbl in LABELS}
    for rid, orig in clean_by_id.items():
        p = pert_by_id.get(rid)
        if p is None:
            continue
        gold = orig["gold_label"]
        if gold not in LABELS:
            continue
        totals[gold] += 1
        if orig["pred_label"] == gold and p["pred_label"] != gold:
            counts[gold] += 1
    return {lbl: {"degraded": counts[lbl], "total": totals[lbl],
                  "rate": round(counts[lbl]/totals[lbl], 4) if totals[lbl] else 0}
            for lbl in LABELS}


def main():
    mcnemar_rows   = []
    ci_rows        = []
    label_deg_rows = []

    for model in MODELS:
        clean_recs = load_outputs(f"outputs/{model}_clean.jsonl")
        if clean_recs is None:
            print(f"  [skip] {model}_clean not found")
            continue

        # Bootstrap CI for clean
        mean_f1, lo, hi = bootstrap_macro_f1(clean_recs)
        ci_rows.append({
            "model": model, "variant": "clean",
            "macro_f1_mean": mean_f1,
            "ci_95_lower": lo, "ci_95_upper": hi
        })

        for variant in VARIANTS:
            pert_recs = load_outputs(f"outputs/{model}_{variant}.jsonl")
            if pert_recs is None:
                continue

            # McNemar
            mn = mcnemar_test(clean_recs, pert_recs)
            mcnemar_rows.append({"model": model, "variant": variant, **mn})

            # Bootstrap CI for perturbed
            mean_f1, lo, hi = bootstrap_macro_f1(pert_recs)
            ci_rows.append({
                "model": model, "variant": variant,
                "macro_f1_mean": mean_f1,
                "ci_95_lower": lo, "ci_95_upper": hi
            })

            # Per-label degradation
            deg = label_degradation(clean_recs, pert_recs)
            for lbl, d in deg.items():
                label_deg_rows.append({
                    "model": model, "variant": variant, "gold_label": lbl,
                    **d
                })

    # Save McNemar
    mn_df = pd.DataFrame(mcnemar_rows)
    mn_df.to_csv("scores/aggregate/mcnemar_results.csv", index=False)
    print("Saved: scores/aggregate/mcnemar_results.csv")
    print(mn_df.to_string(index=False))

    # Save CI
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv("scores/aggregate/bootstrap_ci.csv", index=False)
    print("\nSaved: scores/aggregate/bootstrap_ci.csv")

    # Save label degradation
    ld_df = pd.DataFrame(label_deg_rows)
    ld_df.to_csv("scores/aggregate/label_degradation.csv", index=False)
    print("Saved: scores/aggregate/label_degradation.csv")

    # Failure case counts
    if os.path.exists("scores/aggregate/failure_cases.jsonl"):
        fc_counts = {}
        with open("scores/aggregate/failure_cases.jsonl") as f:
            for line in f:
                c = json.loads(line)
                key = (c["model"], c["variant"], c["type"])
                fc_counts[key] = fc_counts.get(key, 0) + 1
        fc_rows = [{"model": k[0], "variant": k[1], "type": k[2], "count": v}
                   for k, v in sorted(fc_counts.items())]
        fc_df = pd.DataFrame(fc_rows)
        fc_df.to_csv("scores/aggregate/failure_case_counts.csv", index=False)
        print("Saved: scores/aggregate/failure_case_counts.csv")

    print("\nStatistical analysis complete.")


if __name__ == "__main__":
    main()
