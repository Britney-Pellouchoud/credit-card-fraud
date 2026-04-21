import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# STYLE (paper-friendly)
# -----------------------------
sns.set_style("whitegrid")

# -----------------------------
# PATHS (robust + no nesting bugs)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "paper_results.csv"
FIG_DIR = PROJECT_ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("\n📊 Loaded results:")
print(df.head())

# -----------------------------
# CLEAN LABELING
# -----------------------------
df["is_ga"] = df["vector"].astype(str).str.lower() != "full"

# -----------------------------
# SAFE SUMMARY STATS (IMPORTANT FOR PAPER)
# -----------------------------
summary = df.groupby("is_ga")[["auc", "f1", "precision", "recall", "accuracy"]].mean()
print("\n📊 Mean performance (GA vs FULL):")
print(summary)

summary.to_csv(FIG_DIR / "summary_stats.csv")

# -----------------------------
# HELPERS
# -----------------------------
def save_fig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 saved {path}")

# -----------------------------
# 1. AUC COMPARISON
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="is_ga", y="auc")
plt.xticks([0, 1], ["Full Model", "GA Selected"])
plt.title("AUC Distribution: GA vs Full Feature Set")
save_fig("auc_comparison.png")

# -----------------------------
# 2. F1 COMPARISON
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="is_ga", y="f1")
plt.xticks([0, 1], ["Full Model", "GA Selected"])
plt.title("F1 Distribution: GA vs Full Feature Set")
save_fig("f1_comparison.png")

# -----------------------------
# 3. MODEL BREAKDOWN
# -----------------------------
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="model", y="auc", hue="is_ga")
plt.title("AUC by Model: GA vs Full")
save_fig("model_auc.png")

# -----------------------------
# 4. GA IMPACT (IMPORTANT PLOT)
#    → THIS is what makes your paper stronger
# -----------------------------
ga_df = df[df["is_ga"]].copy()
full_df = df[df["is_ga"] == False].copy()

# average FULL baseline per model
full_means = full_df.groupby("model")["auc"].mean().reset_index()
ga_means = ga_df.groupby("model")["auc"].mean().reset_index()

delta = pd.merge(full_means, ga_means, on="model", suffixes=("_full", "_ga"))
delta["auc_improvement"] = delta["auc_ga"] - delta["auc_full"]

plt.figure(figsize=(7, 4))
sns.barplot(data=delta, x="model", y="auc_improvement")
plt.axhline(0, color="black", linewidth=1)
plt.title("GA Impact on AUC (Δ vs Full Features)")
plt.ylabel("Δ AUC (GA - Full)")
save_fig("ga_delta_auc.png")

# -----------------------------
# 5. BEST / WORST GA RESULTS
# -----------------------------
best = ga_df.sort_values("auc", ascending=False).head(1)
worst = ga_df.sort_values("auc", ascending=True).head(1)

print("\n🏆 BEST GA RESULT:")
print(best)

print("\n⚠️ WORST GA RESULT:")
print(worst)

# save readable report
with open(FIG_DIR / "ga_summary.txt", "w") as f:
    f.write("=== BEST GA RESULT ===\n")
    f.write(best.to_string(index=False))
    f.write("\n\n=== WORST GA RESULT ===\n")
    f.write(worst.to_string(index=False))

# -----------------------------
# FINAL MESSAGE
# -----------------------------
print(f"\n✅ All figures saved to: {FIG_DIR}")