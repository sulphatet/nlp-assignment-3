"""
Script 05: Compute fairness / disaggregated bias metrics.

Analyses:
  A. AAVE FPR analysis — False Positive Rate on AAVE-associated normal posts
  B. Identity mention bias — predicted hate probability on neutral identity sentences
  C. Counterfactual symmetry — confidence delta for demographic swap (variant: demo_swap)
  D. Register shift recall — hate_speech recall drop on formal register variant

Outputs:
  scores/fairness/{model}_fairness.json
  scores/fairness/aave_fpr.csv
  scores/fairness/identity_mentions.csv
  scores/fairness/cda_symmetry.csv
"""
import os, json, itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, fisher_exact
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

os.makedirs("scores/fairness", exist_ok=True)

MODELS   = ["bert_hatexplain", "twitter_roberta", "dynabench_roberta"]
LABELS   = ["hate_speech", "offensive", "normal"]

# ── A. Minimal AAVE word list (proxy; from Sap et al. 2019 Appendix) ──────────
AAVE_MARKERS = {
    "finna", "imma", "tryna", "bout", "ion", "ngl", "lowkey", "highkey",
    "deadass", "fasho", "on god", "no cap", "salty", "litty", "woke",
    "shook", "extra", "periodt", "bussin", "sus", "vibe", "clout",
    "drip", "goat", "snack", "flex", "slay", "bet", "fr fr",
    "bruh", "sis", "bae", "ratchet", "twerk", "boo",
}

# ── B. Identity mention templates ─────────────────────────────────────────────
IDENTITY_GROUPS = {
    "Muslim":      ["Muslim", "Islamic", "Islam"],
    "Jewish":      ["Jewish", "Jewish person", "Judaism"],
    "Christian":   ["Christian", "Christianity"],
    "Black":       ["Black", "African American"],
    "White":       ["White", "Caucasian"],
    "Gay":         ["gay", "homosexual", "LGBTQ"],
    "Straight":    ["straight", "heterosexual"],
    "Women":       ["women", "female", "girls"],
    "Men":         ["men", "male", "boys"],
}

TEMPLATES = [
    "I am {term}.",
    "My friend is {term}.",
    "Many {term} people live in this city.",
    "She is {term}.",
    "He is {term}.",
]

def load_outputs(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


# ── A. AAVE FPR ────────────────────────────────────────────────────────────────
def is_aave(text):
    text_lower = text.lower()
    return any(marker in text_lower for marker in AAVE_MARKERS)

def aave_fpr_analysis(model_name, clean_records):
    """Compute FPR for AAVE-associated normal posts vs. SAE normal posts."""
    normal_records = [r for r in clean_records if r["gold_label"] == "normal"]

    aave_recs = [r for r in normal_records if is_aave(r["text"])]
    sae_recs  = [r for r in normal_records if not is_aave(r["text"])]

    def fpr(recs):
        if not recs:
            return None, 0, 0
        fp = sum(1 for r in recs if r["pred_label"] != "normal")
        return round(fp / len(recs), 4), fp, len(recs)

    aave_fpr, aave_fp, aave_n = fpr(aave_recs)
    sae_fpr,  sae_fp,  sae_n  = fpr(sae_recs)

    # Fisher's exact test on 2×2 contingency table: [[TP, FP], [TN, FN]] approximate
    if aave_n > 0 and sae_n > 0:
        contingency = [
            [aave_n - aave_fp, aave_fp],
            [sae_n  - sae_fp,  sae_fp],
        ]
        _, p_value = fisher_exact(contingency)
    else:
        p_value = None

    return {
        "model":     model_name,
        "aave_n":    aave_n,
        "aave_fpr":  aave_fpr,
        "sae_n":     sae_n,
        "sae_fpr":   sae_fpr,
        "fpr_ratio": round(aave_fpr / sae_fpr, 3) if sae_fpr and aave_fpr else None,
        "p_value":   round(p_value, 4) if p_value is not None else None,
        "significant_alpha_05": bool(p_value is not None and p_value < 0.05),
    }


# ── B. Identity mention bias ────────────────────────────────────────────────────
def identity_mention_analysis(model_name, clf_fn):
    """
    Build neutral identity sentences and measure predicted hate_speech probability.
    clf_fn: function(list[str]) -> list[dict with prob_hate_speech etc.]
    """
    rows = []
    for group, terms in IDENTITY_GROUPS.items():
        for term in terms:
            for template in TEMPLATES:
                text = template.format(term=term)
                rows.append({"group": group, "term": term, "text": text})

    df = pd.DataFrame(rows)
    return df


# ── C. Counterfactual Symmetry (CDA) ─────────────────────────────────────────
def cda_symmetry_analysis(model_name, clean_records, swap_records):
    """
    Compare P(hate_speech) between original and demographically-swapped text.
    Expects clean_records and swap_records aligned (same IDs, same order).
    """
    deltas = []
    for orig, swap in zip(clean_records, swap_records):
        assert orig["id"] == swap["id"], "ID mismatch between clean and swap datasets"
        delta = orig["prob_hate_speech"] - swap["prob_hate_speech"]
        deltas.append({
            "id":          orig["id"],
            "gold_label":  orig["gold_label"],
            "orig_prob":   orig["prob_hate_speech"],
            "swap_prob":   swap["prob_hate_speech"],
            "delta":       round(delta, 6),
            "abs_delta":   round(abs(delta), 6),
        })

    df = pd.DataFrame(deltas)
    abs_deltas = df["abs_delta"].values

    # Wilcoxon signed-rank test: is the distribution of deltas symmetric around 0?
    try:
        stat, p = wilcoxon(df["delta"].values)
    except Exception:
        stat, p = None, None

    summary = {
        "model":              model_name,
        "mean_abs_delta":     round(float(np.mean(abs_deltas)), 4),
        "median_abs_delta":   round(float(np.median(abs_deltas)), 4),
        "pct_changed_gt_01":  round(float(np.mean(abs_deltas > 0.1)), 4),
        "wilcoxon_stat":      round(float(stat), 4) if stat is not None else None,
        "wilcoxon_p":         round(float(p), 6) if p is not None else None,
        "significant_alpha_05": bool(p is not None and p < 0.05),
    }
    return summary, df


# ── D. Register shift recall ──────────────────────────────────────────────────
def register_recall_analysis(model_name, clean_records, reg_records):
    def hate_recall(recs):
        hate_golds = [r for r in recs if r["gold_label"] == "hate_speech"]
        if not hate_golds:
            return None
        correct = sum(1 for r in hate_golds if r["pred_label"] == "hate_speech")
        return round(correct / len(hate_golds), 4)

    return {
        "model":          model_name,
        "clean_recall":   hate_recall(clean_records),
        "register_recall": hate_recall(reg_records),
    }


def main():
    aave_rows    = []
    cda_summaries = []
    reg_rows     = []
    id_rows      = []

    for model in MODELS:
        clean_path = f"outputs/{model}_clean.jsonl"
        swap_path  = f"outputs/{model}_demo_swap.jsonl"
        reg_path   = f"outputs/{model}_register_shift.jsonl"

        if not os.path.exists(clean_path):
            print(f"  [skip] {clean_path} not found")
            continue

        clean_records = load_outputs(clean_path)

        # A. AAVE FPR
        print(f"  Running AAVE FPR for {model}...")
        aave_result = aave_fpr_analysis(model, clean_records)
        aave_rows.append(aave_result)

        # B. Identity mention sentences (text only, inference happens in run_models)
        id_df = identity_mention_analysis(model, clf_fn=None)
        # Save texts so script 03 can pick them up after we add a variant
        id_df.to_csv(f"scores/fairness/identity_sentences_{model}.csv", index=False)

        # C. CDA symmetry
        if os.path.exists(swap_path):
            print(f"  Running CDA symmetry for {model}...")
            swap_records = load_outputs(swap_path)
            summary, detail_df = cda_symmetry_analysis(model, clean_records, swap_records)
            cda_summaries.append(summary)
            detail_df.to_csv(f"scores/fairness/cda_detail_{model}.csv", index=False)
        else:
            print(f"  [skip] CDA: {swap_path} not found")

        # D. Register recall
        if os.path.exists(reg_path):
            print(f"  Running register recall for {model}...")
            reg_records = load_outputs(reg_path)
            reg_row = register_recall_analysis(model, clean_records, reg_records)
            reg_rows.append(reg_row)
        else:
            print(f"  [skip] Register: {reg_path} not found")

    # Save aggregate fairness tables
    if aave_rows:
        pd.DataFrame(aave_rows).to_csv("scores/fairness/aave_fpr.csv", index=False)
        print("Saved: scores/fairness/aave_fpr.csv")
    if cda_summaries:
        pd.DataFrame(cda_summaries).to_csv("scores/fairness/cda_symmetry.csv", index=False)
        print("Saved: scores/fairness/cda_symmetry.csv")
    if reg_rows:
        pd.DataFrame(reg_rows).to_csv("scores/fairness/register_recall.csv", index=False)
        print("Saved: scores/fairness/register_recall.csv")

    print("\nFairness analysis complete.")

if __name__ == "__main__":
    main()
