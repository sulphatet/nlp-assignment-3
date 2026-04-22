"""
Script 00: Download HateXplain and create stratified 900-example test set.

Downloads directly from the official HateXplain GitHub repository:
https://github.com/hate-alert/HateXplain

Produces: data/clean/hatexplain_test_900.jsonl
"""
import os
import json
import random
import urllib.request
import numpy as np
import pandas as pd
from collections import Counter

# Deterministic seed
random.seed(42)
np.random.seed(42)

DATASET_URL  = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
SPLITS_URL   = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)

def download(url, dest):
    if not os.path.exists(dest):
        print(f"  Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"  Using cached {dest}")

def majority_label(annotators):
    labels = [a["label"] for a in annotators]
    c = Counter(labels)
    return c.most_common(1)[0][0]

def target_communities(annotators):
    targets = set()
    for a in annotators:
        for t in a.get("target", []):
            if t and t.lower() != "none":
                targets.add(t)
    return sorted(targets)

def main():
    print("Downloading HateXplain dataset...")
    download(DATASET_URL, "data/raw/dataset.json")
    download(SPLITS_URL,  "data/raw/post_id_divisions.json")

    with open("data/raw/dataset.json")  as f: dataset   = json.load(f)
    with open("data/raw/post_id_divisions.json") as f: splits = json.load(f)

    test_ids = set(splits["test"])
    print(f"Test IDs in official split: {len(test_ids)}")

    label_map = {"hatespeech": "hate_speech", "normal": "normal", "offensive": "offensive"}

    records = []
    for post_id, item in dataset.items():
        if post_id not in test_ids:
            continue
        raw_label = majority_label(item["annotators"])
        label     = label_map.get(raw_label, raw_label)
        text      = " ".join(item["post_tokens"])
        targets   = target_communities(item["annotators"])
        records.append({
            "id":     post_id,
            "text":   text,
            "label":  label,
            "target": targets,
        })

    df = pd.DataFrame(records)
    print(f"Total test examples: {len(df)}")
    print(df["label"].value_counts().to_dict())

    # Stratified sample: up to 300 per class
    sampled = (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), 300), random_state=42))
    )
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sampled)} examples: {sampled['label'].value_counts().to_dict()}")

    out = "data/clean/hatexplain_test_900.jsonl"
    sampled.to_json(out, orient="records", lines=True)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
