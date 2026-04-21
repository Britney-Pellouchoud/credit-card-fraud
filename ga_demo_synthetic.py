import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from training_alg.with_ga.paper_ga import PaperGA


# -----------------------------
# PATH SETUP
# -----------------------------
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# HELPER: SAVE FIG
# -----------------------------
def save_fig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"💾 saved {path}")


# -----------------------------
# 1. CREATE SYNTHETIC DATA
# -----------------------------
print("\n🧪 Creating synthetic dataset...")

X, y = make_classification(
    n_samples=5000,
    n_features=50,
    n_informative=5,
    n_redundant=5,
    random_state=42,
    shuffle=False
)

X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Total features: {X.shape[1]}")


# -----------------------------
# 2. BASELINE (ALL FEATURES)
# -----------------------------
print("\n🚫 Running baseline...")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

baseline_pred = rf.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

print(f"Baseline Accuracy: {baseline_acc:.4f}")


# -----------------------------
# 3. GA FEATURE SELECTION
# -----------------------------
print("\n🧬 Running GA feature selection...")

ga = PaperGA(
    pop_size=15,
    generations=8,
    mutation_rate=0.05,
    random_state=42
)

vectors = ga.run_multi_vectors(X_train, y_train, n_vectors=1)
selected_features = vectors[0]

print(f"\n✅ Selected {len(selected_features)} features")
print(selected_features)


# -----------------------------
# 4. GA MODEL
# -----------------------------
print("\n🧬 Evaluating GA-selected features...")

rf_ga = RandomForestClassifier(random_state=42)
rf_ga.fit(X_train[selected_features], y_train)

ga_pred = rf_ga.predict(X_test[selected_features])
ga_acc = accuracy_score(y_test, ga_pred)

print(f"GA Accuracy: {ga_acc:.4f}")


# -----------------------------
# 5. COMPARISON PLOT
# -----------------------------
print("\n📊 Creating comparison plots...")

results_df = pd.DataFrame({
    "method": ["Baseline (All Features)", "GA (Selected Features)"],
    "accuracy": [baseline_acc, ga_acc],
    "n_features": [X.shape[1], len(selected_features)]
})

# Accuracy comparison
plt.figure()
plt.bar(results_df["method"], results_df["accuracy"])
plt.title("Accuracy Comparison: GA vs Baseline")
plt.ylabel("Accuracy")
save_fig("ga_vs_baseline_accuracy.png")

# Feature reduction
plt.figure()
plt.bar(results_df["method"], results_df["n_features"])
plt.title("Feature Count Reduction")
plt.ylabel("Number of Features")
save_fig("ga_feature_reduction.png")


# -----------------------------
# 6. SUMMARY PRINT
# -----------------------------
print("\n📊 FINAL COMPARISON")
print(results_df)

print("\n📈 Improvement:")
print(f"Δ Accuracy: {ga_acc - baseline_acc:.4f}")
print(f"Feature Reduction: {X.shape[1]} → {len(selected_features)}")

print(f"\n✅ Figures saved in: {FIG_DIR}")