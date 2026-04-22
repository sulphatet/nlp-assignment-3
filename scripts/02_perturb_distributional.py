import os
import json
import random
import numpy as np
import spacy
import copy
import contractions
import re
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

# Lexicons and Mappings
INFORMAL_TO_FORMAL = {
    "tbh": "to be honest",
    "ngl": "not going to lie",
    "fr": "for real",
    "idk": "I do not know",
    "lol": "amusing",
    "omg": "my goodness",
    "wtf": "what the hell",
    "smh": "shaking my head",
    "gonna": "going to",
    "wanna": "want to",
    "bout": "about",
    "ain't": "is not",
    "yall": "you all",
    "y'all": "you all",
    "cuz": "because",
    "cause": "because"
}

EN_HI_DICT = {
    "people": "log",
    "hate": "nafrat",
    "love": "pyar",
    "today": "aaj",
    "you": "tum",
    "me": "mujhe",
    "they": "ve",
    "brother": "bhai",
    "sister": "behen",
    "time": "waqt",
    "what": "kya",
    "why": "kyon",
    "good": "accha",
    "bad": "kharab",
    "yes": "haan",
    "no": "nahi",
    "very": "bohut",
    "friend": "dost",
    "money": "paisa",
    "world": "duniya",
    "all": "sab",
    "life": "zindagi",
    "big": "bada",
    "small": "cchota"
}

SWAP_DICT = {
    "muslim": "christian",
    "muslims": "christians",
    "islam": "christianity",
    "islamic": "christian",
    "christian": "muslim",
    "christians": "muslims",
    "christianity": "islam",
    
    "black": "white",
    "african": "caucasian",
    "white": "black",
    "caucasian": "african",
    
    "gay": "straight",
    "homosexual": "heterosexual",
    "straight": "gay",
    "heterosexual": "homosexual",
    "lgbtq": "heterosexual",
    
    "women": "men",
    "woman": "man",
    "girl": "boy",
    "girls": "boys",
    "female": "male",
    "men": "women",
    "man": "woman",
    "boy": "girl",
    "boys": "girls",
    "male": "female",
    
    "jewish": "christian",
    "jew": "christian",
    "jews": "christians",
    "judaism": "christianity"
}

def apply_register_shift(data):
    perturbed = copy.deepcopy(data)
    for item in tqdm(perturbed, desc="Applying Register Shift"):
        text = item['text']
        # Expand contractions first
        try:
            text = contractions.fix(text)
        except Exception:
            pass
        # Replace informal slang
        tokens = text.split()
        new_tokens = []
        for token in tokens:
            lower_token = token.lower().strip(".,!?\"'")
            if lower_token in INFORMAL_TO_FORMAL:
                # Maintain roughly the same case if possible, simplify for now
                replacement = INFORMAL_TO_FORMAL[lower_token]
                new_tokens.append(replacement)
            else:
                new_tokens.append(token)
        item['text'] = " ".join(new_tokens)
    return perturbed

def apply_code_mixing(data, nlp):
    perturbed = copy.deepcopy(data)
    for item in tqdm(perturbed, desc="Applying Code Mixing (En-Hi)"):
        doc = nlp(item['text'])
        new_tokens = []
        for token in doc:
            word = token.text.lower()
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET'] and word in EN_HI_DICT:
                # 50% chance to replace if in dictionary to avoid full translation
                if random.random() < 0.5:
                    rep = EN_HI_DICT[word]
                    # Attempt to match capitalization
                    if token.text.istitle():
                        rep = rep.capitalize()
                    elif token.text.isupper():
                        rep = rep.upper()
                    new_tokens.append(rep + token.whitespace_)
                else:
                    new_tokens.append(token.text_with_ws)
            else:
                new_tokens.append(token.text_with_ws)
        item['text'] = "".join(new_tokens).strip()
    return perturbed

def apply_demographic_swap(data, nlp):
    perturbed = copy.deepcopy(data)
    for item in tqdm(perturbed, desc="Applying Demographic Swap"):
        # We perform simple substring replacement via regex on whole words
        text = item['text']
        for k, v in SWAP_DICT.items():
            # Use regex to match whole words and replace (case-insensitive, preserving case roughly)
            pattern = re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE)
            def matchcase(word):
                def replace(m):
                    target = m.group(0)
                    if target.isupper():
                        return word.upper()
                    elif target.istitle():
                        return word.capitalize()
                    elif target.islower():
                        return word.lower()
                    return word
                return replace
            text = pattern.sub(matchcase(v), text)
            
        item['text'] = text.strip()
    return perturbed


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

    
    print("Generating C1: Register Shift")
    register_data = apply_register_shift(data)
    save_data(register_data, "data/distributional/register_shift.jsonl")
    
    print("Generating C2: Code-mixed")
    code_mixed_data = apply_code_mixing(data, nlp)
    save_data(code_mixed_data, "data/distributional/code_mixed.jsonl")
    
    print("Generating C3: Demographic swap")
    demo_swap_data = apply_demographic_swap(data, nlp)
    save_data(demo_swap_data, "data/distributional/demographic_swap.jsonl")
    
    print("All distributional perturbations saved.")

if __name__ == "__main__":
    main()
