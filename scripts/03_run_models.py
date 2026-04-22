"""
Script 03: Run inference with 3 hate-speech models on all dataset variants.

Models:
  1. Hate-speech-CNERG/bert-base-uncased-hatexplain   (BERT-HateXplain)
  2. cardiffnlp/twitter-roberta-base-hate-latest       (Twitter-RoBERTa)
  3. facebook/roberta-hate-speech-dynabench-r4-target  (DynaBench-RoBERTa)

For each model × variant, saves a .jsonl with:
  {id, text, gold_label, pred_label, prob_hate, prob_offensive, prob_normal}
"""
import os, json, random
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = 0 if torch.cuda.is_available() else -1

# ── Model registry ──────────────────────────────────────────────────────────────
# Each entry: (short_name, hf_id, label_remap)
# label_remap maps the model's raw output label strings to our canonical set:
#   hate_speech | offensive | normal
MODELS = [
    (
        "bert_hatexplain",
        "Hate-speech-CNERG/bert-base-uncased-hatexplain",
        {"hate speech": "hate_speech", "offensive language": "offensive", "normal": "normal"},
    ),
    (
        "twitter_roberta",
        "cardiffnlp/twitter-roberta-base-hate-latest",
        # This model outputs: hate / non-hate (binary) — we map to our 3-class via threshold
        # We store raw probs and treat "non-hate" as normal; offensive is approximated later.
        {"hate": "hate_speech", "non-hate": "normal"},
    ),
    (
        "dynabench_roberta",
        "facebook/roberta-hate-speech-dynabench-r4-target",
        # outputs: hate / nothate
        {"hate": "hate_speech", "nothate": "normal"},
    ),
]

VARIANTS = {
    "clean":           "data/clean/hatexplain_test_900.jsonl",
    "typos":           "data/surface_perturbed/typos.jsonl",
    "paraphrase":      "data/surface_perturbed/paraphrase.jsonl",
    "word_order":      "data/surface_perturbed/word_order.jsonl",
    "combined":        "data/surface_perturbed/combined.jsonl",
    "register_shift":  "data/distributional/register_shift.jsonl",
    "code_mixed":      "data/distributional/code_mixed.jsonl",
    "demo_swap":       "data/distributional/demographic_swap.jsonl",
}

os.makedirs("outputs", exist_ok=True)

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_outputs(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def run_on_all_variants():
    for model_name, hf_id, label_remap in MODELS:
        # Check if we need to load the model at all
        any_missing = False
        for variant_name in VARIANTS:
            if not os.path.exists(f"outputs/{model_name}_{variant_name}.jsonl"):
                any_missing = True
                break
        
        if not any_missing:
            print(f"  [skip] All outputs for {model_name} already exist")
            continue

        print(f"\n{'='*60}")
        print(f"Loading Model: {model_name} ({hf_id})")
        print(f"{'='*60}")
        
        clf = pipeline(
            "text-classification",
            model=hf_id,
            tokenizer=hf_id,
            top_k=None,
            truncation=True,
            max_length=512,
            device=DEVICE,
        )

        for variant_name, variant_path in VARIANTS.items():
            out_path = f"outputs/{model_name}_{variant_name}.jsonl"
            if os.path.exists(out_path):
                print(f"  [skip] {out_path} already exists")
                continue
            if not os.path.exists(variant_path):
                print(f"  [skip] {variant_path} not found")
                continue

            print(f"  Running variant: {variant_name}")
            data = load_data(variant_path)
            batch_texts = [item["text"] for item in data]
            BATCH = 32

            all_preds = []
            for i in tqdm(range(0, len(batch_texts), BATCH), desc=f"    {model_name}"):
                batch = batch_texts[i:i+BATCH]
                out = clf(batch)
                all_preds.extend(out)

            results = []
            for item, preds in zip(data, all_preds):
                prob_map = {p["label"].lower(): p["score"] for p in preds}
                canonical = {"hate_speech": 0.0, "offensive": 0.0, "normal": 0.0}
                for raw_lbl, canon_lbl in label_remap.items():
                    score = prob_map.get(raw_lbl, prob_map.get(raw_lbl.replace(" ", "_"), 0.0))
                    canonical[canon_lbl] = max(canonical[canon_lbl], score)

                total = sum(canonical.values())
                if total < 0.99 and canonical["offensive"] == 0.0:
                    canonical["offensive"] = max(0.0, 1.0 - total)

                pred_label = max(canonical, key=canonical.get)
                results.append({
                    "id":           item["id"],
                    "text":         item["text"],
                    "gold_label":   item["label"],
                    "pred_label":   pred_label,
                    "prob_hate_speech": round(canonical["hate_speech"], 6),
                    "prob_offensive":   round(canonical["offensive"], 6),
                    "prob_normal":      round(canonical["normal"], 6),
                    "target":       item.get("target", []),
                })
            
            save_outputs(results, out_path)
            print(f"    Saved records → {out_path}")

def main():
    run_on_all_variants()
    print("\nAll inference complete.")

if __name__ == "__main__":
    main()
