import os
import json
import random
import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import spacy
import copy
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def apply_typos(data):
    # Keyboard augmenter simulates typing errors
    aug = nac.KeyboardAug(aug_char_p=0.15, aug_word_p=0.15)
    perturbed = copy.deepcopy(data)
    for item in tqdm(perturbed, desc="Applying Typos"):
        try:
            res = aug.augment(item['text'])
            item['text'] = res[0] if isinstance(res, list) else res
        except Exception:
            pass
    return perturbed

def apply_paraphrase(data):
    # Synonym substitution using WordNet
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
    perturbed = copy.deepcopy(data)
    for item in tqdm(perturbed, desc="Applying Paraphrase"):
        try:
            res = aug.augment(item['text'])
            item['text'] = res[0] if isinstance(res, list) else res
        except Exception:
            pass
    return perturbed

def apply_word_order(data, nlp):
    perturbed = copy.deepcopy(data)
    for item in tqdm(perturbed, desc="Applying Word Order Shift"):
        doc = nlp(item['text'])
        new_tokens = []
        i = 0
        while i < len(doc):
            # Adjective - Noun swap
            if (i < len(doc) - 1
                    and doc[i].pos_ == 'ADJ'
                    and doc[i + 1].pos_ == 'NOUN'):
                # Swap: noun then adjective, keeping whitespace of the adjective
                new_tokens.append(doc[i + 1].text + doc[i].whitespace_)
                new_tokens.append(doc[i].text + doc[i + 1].whitespace_)
                i += 2
            else:
                new_tokens.append(doc[i].text_with_ws)
                i += 1
        item['text'] = "".join(new_tokens).strip()
    return perturbed

def apply_combined(data, nlp):
    # All three sequentially
    res1 = apply_word_order(data, nlp)
    res2 = apply_paraphrase(res1)
    res3 = apply_typos(res2)
    return res3

def main():
    print("Loading clean data...")
    clean_path = "data/clean/hatexplain_test_900.jsonl"
    if not os.path.exists(clean_path):
        print(f"File {clean_path} not found. Run script 00 first.")
        return
        
    data = load_data(clean_path)
    
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    
    print("Generating B1: Typos")
    typos_data = apply_typos(data)
    save_data(typos_data, "data/surface_perturbed/typos.jsonl")
    
    print("Generating B2: Paraphrase")
    paraphrase_data = apply_paraphrase(data)
    save_data(paraphrase_data, "data/surface_perturbed/paraphrase.jsonl")
    
    print("Generating B3: Word order")
    word_order_data = apply_word_order(data, nlp)
    save_data(word_order_data, "data/surface_perturbed/word_order.jsonl")
    
    print("Generating B4: Combined")
    combined_data = apply_combined(data, nlp)
    save_data(combined_data, "data/surface_perturbed/combined.jsonl")
    
    print("All surface perturbations saved.")

if __name__ == "__main__":
    main()
