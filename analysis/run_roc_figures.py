import os
import json
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from training_alg.with_ga.paper_feature_vectors import VECTORS
from analysis.roc_paper_figures import plot_roc
from config import RF_ESTIMATORS_FINAL, RUN_ONLY_ONE_VECTOR


def run_figures(X_train, X_test, y_train, y_test, output_dir="outputs/figures"):
    """
    Generates ROC curves (Paper Figures 4–8 equivalent)
    and saves AUC results to disk.
    """

    print("\nSTARTING ROC FIGURES (Paper Figs 4–8)\n")

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    vectors = list(VECTORS.items())

    if RUN_ONLY_ONE_VECTOR:
        print("⚠ RUN_ONLY_ONE_VECTOR=True → only running v1")
        vectors = vectors[:1]

    for i, (name, cols) in enumerate(vectors, start=4):

        print(f"\n--- Running {name} (Figure {i}) ---")

        # safety check (prevents silent empty feature crashes)
        cols = [c for c in cols if c in X_train.columns]

        if len(cols) == 0:
            print(f"⚠ WARNING: {name} has no valid columns. Skipping.")
            continue

        X_tr = X_train[cols]
        X_te = X_test[cols]

        print(f"Using {len(cols)} features")

        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS_FINAL,
            random_state=42,
            n_jobs=-1
        )

        print("Training Random Forest...")

        auc_score = plot_roc(
            model,
            X_tr, X_te,
            y_train, y_test,
            name,
            i
        )

        print(f"{name} AUC: {auc_score:.4f}")

        results[name] = {
            "AUC": float(auc_score),
            "n_features": len(cols)
        }

    print("\nALL ROC FIGURES COMPLETE\n")

    # --------------------------
    # SAVE OUTPUTS (IMPORTANT FIX)
    # --------------------------
    csv_path = os.path.join(output_dir, "roc_auc_summary.csv")
    json_path = os.path.join(output_dir, "roc_auc_summary.json")

    df = pd.DataFrame([
        {"Vector": k, **v} for k, v in results.items()
    ])

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved ROC summary CSV → {csv_path}")
    print(f"Saved ROC summary JSON → {json_path}")

    return results