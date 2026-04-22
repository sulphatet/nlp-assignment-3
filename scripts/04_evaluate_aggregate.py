"""
Script 04: Compute aggregate classification metrics for every model × variant.

Metrics computed:
  - Accuracy
  - Per-class Precision, Recall, F1
  - Macro-F1, Weighted-F1
  - AUC-ROC (one-vs-rest)
  - Confusion matrix

Outputs:
  scores/aggregate/{model}_{variant}_metrics.json
  scores/aggregate/summary_table.csv
"""
import os, json, glob
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, roc_auc_score, confusion_matrix
)

LABELS = ["hate_speech", "offensive", "normal"]
MODELS   = ["bert_hatexplain", "twitter_roberta", "dynabench_roberta"]
VARIANTS = ["clean", "typos", "paraphrase", "word_order", "combined",
            "register_shift", "code_mixed", "demo_swap"]

os.makedirs("scores/aggregate", exist_ok=True)

def load_outputs(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def compute_metrics(records):
    golds = [r["gold_label"] for r in records]
    preds = [r["pred_label"] for r in records]

    # Probability matrix for AUC [N, 3]
    prob_matrix = np.array([
        [r["prob_hate_speech"], r["prob_offensive"], r["prob_normal"]]
        for r in records
    ])

    accuracy  = accuracy_score(golds, preds)
    macro_f1  = f1_score(golds, preds, average="macro",    labels=LABELS, zero_division=0)
    weighted_f1 = f1_score(golds, preds, average="weighted", labels=LABELS, zero_division=0)

    prec, rec, f1, support = precision_recall_fscore_support(
        golds, preds, labels=LABELS, zero_division=0
    )

    per_class = {}
    for i, lbl in enumerate(LABELS):
        per_class[lbl] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(float(f1[i]),   4),
            "support":   int(support[i]),
        }

    # AUC — requires at least 2 classes present
    try:
        # Binarise gold for multiclass OvR
        from sklearn.preprocessing import label_binarize
        gold_bin = label_binarize(golds, classes=LABELS)
        auc = roc_auc_score(gold_bin, prob_matrix, multi_class="ovr", average="macro")
        auc = round(float(auc), 4)
    except Exception:
        auc = None

    cm = confusion_matrix(golds, preds, labels=LABELS).tolist()

    return {
        "accuracy":    round(accuracy, 4),
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "auc_roc":     auc,
        "per_class":   per_class,
        "confusion_matrix": cm,
        "n":           len(records),
    }

def main():
    summary_rows = []

    for model in MODELS:
        for variant in VARIANTS:
            path = f"outputs/{model}_{variant}.jsonl"
            if not os.path.exists(path):
                print(f"  [skip] {path} not found")
                continue

            records = load_outputs(path)
            metrics = compute_metrics(records)

            out_path = f"scores/aggregate/{model}_{variant}_metrics.json"
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Summary row for the big table
            row = {
                "model":        model,
                "variant":      variant,
                "accuracy":     metrics["accuracy"],
                "macro_f1":     metrics["macro_f1"],
                "weighted_f1":  metrics["weighted_f1"],
                "auc_roc":      metrics["auc_roc"],
                "hate_f1":      metrics["per_class"]["hate_speech"]["f1"],
                "offensive_f1": metrics["per_class"]["offensive"]["f1"],
                "normal_f1":    metrics["per_class"]["normal"]["f1"],
            }
            summary_rows.append(row)
            print(f"  {model} | {variant:15s} | Macro-F1={metrics['macro_f1']:.4f} | Acc={metrics['accuracy']:.4f}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv("scores/aggregate/summary_table.csv", index=False)
        print(f"\nSummary saved to scores/aggregate/summary_table.csv")

if __name__ == "__main__":
    main()
