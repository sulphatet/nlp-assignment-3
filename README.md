# NLP Evaluation Assignment 3: Stress Testing and Robustness

This repository contains the full pipeline for stress testing toxic language detection models.

## Structure
- `data/`: Raw, clean, and perturbed datasets.
- `scripts/`: Implementation scripts (run in numerical order).
- `outputs/`: Raw model predictions (.jsonl).
- `scores/`: Computed metrics and analysis results.
- `report/`: Final report and figures.

## Instructions
1. **Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
   ```

2. **Run Pipeline**:
   ```bash
   # In numerical order:
   python scripts/00_download_and_sample.py
   python scripts/01_perturb_surface.py
   python scripts/02_perturb_distributional.py
   python scripts/03_run_models.py
   python scripts/04_evaluate_aggregate.py
   python scripts/05_evaluate_fairness.py
   python scripts/06_robustness_analysis.py
   python scripts/07_bias_analysis.py
   python scripts/08_plots.py
   ```

3. **View Results**:
   - Final Report: `report/assignment3_report.md`
   - Key Figures: `report/figures/`
   - Metrics Summary: `scores/aggregate/summary_table.csv`
