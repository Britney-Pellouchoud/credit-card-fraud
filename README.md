# 📊 Credit Card Fraud Detection Pipeline (GA + Paper Reproduction)

## Overview

This project implements a full machine learning pipeline for credit card fraud detection, including:
- Genetic Algorithm (GA) based feature selection
- Baseline model evaluation
- Paper-style experimental tables (2–7)
- ROC curve generation (Figures 4–8)
- Fully reproducible experimental runs

The goal is to replicate a research-style pipeline comparing feature selection strategies against full and random feature baselines.

---

## 🔁 End-to-End Pipeline Flow

### 1. Data Loading
The pipeline loads the raw credit card fraud dataset: training_alg/data/raw/creditcard.csv


It then:
- Splits features (X) and target label (y)
- Applies preprocessing (encoding / cleaning)

---

### 2. Train/Test Split
The dataset is split using a stratified split:

- 80% training
- 20% testing
- Stratified to preserve fraud class imbalance

---

### 3. Genetic Algorithm (Feature Selection)

A custom Genetic Algorithm (`PaperGA`) is used to select optimal feature subsets.

Key properties:
- Binary chromosome representation (feature included/excluded)
- Fitness function = Random Forest validation accuracy
- Internal 80/20 split inside fitness evaluation
- Tournament selection
- Single-point crossover
- Random mutation

Outputs:
- 5 evolved feature vectors (v1–v5)
- Convergence history of best fitness per generation

---

### 4. Model Evaluation (Tables 2–7)

Each GA-selected feature set is evaluated using multiple models:

- Random Forest (RF)
- Decision Tree (DT)
- Logistic Regression (LR)
- Naive Bayes (NB)
- Neural Network (MLP)

Metrics computed:
- Accuracy
- Precision
- Recall
- F1-score

Additional baselines:
- Full feature set model
- Randomly selected feature subset

Outputs:
- CSV tables saved per feature vector
- Baseline comparison tables (full vs random vs GA)

---

### 5. ROC Curve Generation (Figures 4–8)

For each GA-selected feature vector:
- A Random Forest classifier is trained
- ROC curve is computed
- AUC is recorded

Outputs:
- ROC plots saved to `outputs/figures/roc/`
- Summary CSV + JSON of AUC scores

---

### 6. Genetic Algorithm Convergence Analysis

The GA tracks:
- Best fitness score per generation
- Convergence behavior over time

Output:
- Convergence plots saved to `outputs/figures/convergence/`

---

## ⚙️ Configuration Modes

The pipeline supports two execution modes:

### Debug Mode
Fast execution for testing:
- Smaller GA population
- Fewer generations
- Reduced dataset size
- Fewer estimators

### Full Paper Mode
Reproducible research setup:
- GA_POP_SIZE = 20
- GA_GENERATIONS = 30
- Full dataset
- Full model evaluation suite

---

## 📦 Outputs

The pipeline generates:
outputs/
├── tables/
│ ├── v1_table.csv
│ ├── v2_table.csv
│ ├── ...
│ ├── full_table.csv
│ └── random_table.csv
├── figures/
│ ├── roc/
│ │ ├── roc curves
│ │ └── auc_summary.json/csv
│ └── convergence/
└── run_summary.json


---

## 🚀 How to Run

Run the full reproduction pipeline:

```bash
python -m orchestration.paper_run
