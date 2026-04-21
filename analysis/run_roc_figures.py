from sklearn.ensemble import RandomForestClassifier
from training_alg.with_ga.paper_feature_vectors import VECTORS
from analysis.roc_paper_figures import plot_roc
from config import RF_ESTIMATORS_FINAL, RUN_ONLY_ONE_VECTOR


def run_figures(X_train, X_test, y_train, y_test):

    print("\nSTARTING ROC FIGURES (Paper Figs 4–8)\n")

    results = {}

    vectors = list(VECTORS.items())

    if RUN_ONLY_ONE_VECTOR:
        vectors = vectors[:1]

    for i, (name, cols) in enumerate(vectors, start=4):

        print(f"\n--- Running {name} (Figure {i}) ---")

        # select feature subset
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

        results[name] = auc_score

    print("\nALL ROC FIGURES COMPLETE\n")

    return results