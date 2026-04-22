"""
Script 06: Robustness analysis — compute Δ-scores and identify metric failure cases.

Reads: scores/aggregate/summary_table.csv  +  individual outputs/*.jsonl
Writes:
  scores/aggregate/delta_table.csv       — Δ (perturbed − clean) per metric × model × variant
  scores/aggregate/failure_cases.jsonl   — sentence-level metric failure examples
"""
import os, json
import numpy as np
import pandas as pd

MODELS   = ["bert_hatexplain", "twitter_roberta", "dynabench_roberta"]
VARIANTS = ["typos", "paraphrase", "word_order", "combined",
            "register_shift", "code_mixed", "demo_swap"]
METRICS  = ["accuracy", "macro_f1", "weighted_f1", "auc_roc",
            "hate_f1", "offensive_f1", "normal_f1"]

def load_outputs(path):
    if not os.path.exists(path):
        return None
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def main():
    summary_path = "scores/aggregate/summary_table.csv"
    if not os.path.exists(summary_path):
        print("Run script 04 first.")
        return

    df = pd.read_csv(summary_path)
    delta_rows = []

    for model in MODELS:
        clean_row = df[(df["model"] == model) & (df["variant"] == "clean")]
        if clean_row.empty:
            continue
        clean_vals = clean_row.iloc[0]

        for variant in VARIANTS:
            pert_row = df[(df["model"] == model) & (df["variant"] == variant)]
            if pert_row.empty:
                continue
            pert_vals = pert_row.iloc[0]

            row = {"model": model, "variant": variant}
            for metric in METRICS:
                try:
                    delta = float(pert_vals[metric]) - float(clean_vals[metric])
                    row[f"delta_{metric}"] = round(delta, 4)
                except (KeyError, TypeError, ValueError):
                    row[f"delta_{metric}"] = None
            delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv("scores/aggregate/delta_table.csv", index=False)
    print("Saved: scores/aggregate/delta_table.csv")
    print(delta_df.to_string(index=False))

    # ── Failure case mining ───────────────────────────────────────────────────
    print("\nMining failure cases…")
    failure_cases = []

    for model in MODELS:
        clean_recs = load_outputs(f"outputs/{model}_clean.jsonl")
        if clean_recs is None:
            continue
        clean_by_id = {r["id"]: r for r in clean_recs}

        for variant in VARIANTS:
            pert_recs = load_outputs(f"outputs/{model}_{variant}.jsonl")
            if pert_recs is None:
                continue
            pert_by_id = {r["id"]: r for r in pert_recs}

            for rec_id, orig in clean_by_id.items():
                pert = pert_by_id.get(rec_id)
                if pert is None:
                    continue

                gold = orig["gold_label"]

                # Failure type 1: Correct on clean → wrong on perturbed (degradation)
                if orig["pred_label"] == gold and pert["pred_label"] != gold:
                    failure_cases.append({
                        "type":       "degradation",
                        "model":      model,
                        "variant":    variant,
                        "id":         rec_id,
                        "gold":       gold,
                        "clean_text": orig["text"],
                        "pert_text":  pert["text"],
                        "clean_pred": orig["pred_label"],
                        "pert_pred":  pert["pred_label"],
                        "clean_prob_hate": orig["prob_hate_speech"],
                        "pert_prob_hate":  pert["prob_hate_speech"],
                    })

                # Failure type 2: Accuracy stable but confidence shifted significantly
                abs_delta = abs(orig["prob_hate_speech"] - pert["prob_hate_speech"])
                if abs_delta > 0.25 and orig["pred_label"] == pert["pred_label"]:
                    failure_cases.append({
                        "type":         "silent_confidence_shift",
                        "model":        model,
                        "variant":      variant,
                        "id":           rec_id,
                        "gold":         gold,
                        "clean_text":   orig["text"],
                        "pert_text":    pert["text"],
                        "same_pred":    orig["pred_label"],
                        "confidence_delta": round(abs_delta, 4),
                    })

    out_path = "scores/aggregate/failure_cases.jsonl"
    with open(out_path, "w") as f:
        for case in failure_cases:
            f.write(json.dumps(case) + "\n")

    print(f"Saved {len(failure_cases)} failure cases → {out_path}")

    # Print a sample of each type for report inspection
    types = set(c["type"] for c in failure_cases)
    for t in types:
        examples = [c for c in failure_cases if c["type"] == t][:2]
        print(f"\n── {t} (showing {len(examples)}) ──")
        for ex in examples:
            print(json.dumps(ex, indent=2))

if __name__ == "__main__":
    main()
