import pandas as pd

from training_alg.data.loader import load_dataset
from training_alg.preprocessing.preprocess import preprocess
from training_alg.with_ga.paper_ga import PaperGA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score, accuracy_score
from config import *


def run():
    import os

    os.makedirs("analysis/figures/roc", exist_ok=True)
    os.makedirs("analysis/figures/convergence", exist_ok=True)

    print("START RUN")

    # --------------------------
    # 1. LOAD + PREPROCESS
    # --------------------------
    print("LOADING DATA")
    X, y = load_dataset("training_alg/data/raw/creditcard.csv")

    print("PREPROCESSING")
    X = preprocess(X)
    if SAMPLE_SIZE:
        print(f"DEBUG MODE: Sampling {SAMPLE_SIZE} rows")
        X = X.sample(SAMPLE_SIZE, random_state=42)
        y = y.loc[X.index]

    # --------------------------
    # 2. SPLIT
    # --------------------------
    print("SPLITTING DATA")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------
    # 3. GA
    # --------------------------
    ga = PaperGA(
        pop_size=GA_POP_SIZE,
        generations=GA_GENERATIONS
    )
    print("ABOUT TO START GA")

    selected = ga.run(X_train, y_train)

    print("GA FINISHED")

    from analysis.ga_convergence import plot_convergence
    plot_convergence(ga.history)

    print("\nSelected features:", len(selected))

    # --------------------------
    # 4. TEST ONE MODEL FIRST (DEBUG STEP)
    # --------------------------
    print("TRAINING SINGLE RF (DEBUG)")

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train.iloc[:, selected], y_train)

    preds = rf.predict(X_test.iloc[:, selected])
    print("RF Accuracy:", accuracy_score(y_test, preds))

    print("SINGLE MODEL DONE")

    # --------------------------
    # 5. FULL PAPER EVALUATION
    # --------------------------
    print("STARTING FULL ROC PIPELINE")

    from analysis.run_roc_figures import run_figures
    run_figures(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("ALL DONE")

    from training_alg.with_ga.paper_feature_vectors import VECTORS
    from analysis.paper_tables import evaluate_models

    print("\nBUILDING PAPER TABLES (2–7)\n")

    all_tables = {}

    for name, cols in VECTORS.items():

        print(f"\n=== Evaluating {name} ===")

        X_tr = X_train[cols]
        X_te = X_test[cols]

        df = evaluate_models(X_tr, X_te, y_train, y_test)

        print(df)

        all_tables[name] = df

    # OPTIONAL: save results
    for name, df in all_tables.items():
        df.to_csv(f"analysis/{name}_table.csv", index=False)

    # FULL FEATURES
    print("\n=== FULL FEATURE VECTOR ===")
    full_df = evaluate_models(X_train, X_test, y_train, y_test)

    # RANDOM VECTOR
    import numpy as np

    random_cols = np.random.choice(X_train.columns, size=20, replace=False)
    print("\n=== RANDOM FEATURE VECTOR ===")
    rand_df = evaluate_models(
        X_train[random_cols],
        X_test[random_cols],
        y_train,
        y_test
    )


if __name__ == "__main__":
    run()