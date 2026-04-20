import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# PATH SETUP
# -----------------------------
ROOT = Path(__file__).resolve().parents[0]
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("paper_results.csv")

print("\n📊 Loaded results:")
print(df.head())

# -----------------------------
# LABEL GA vs FULL
# -----------------------------
df["is_ga"] = df["vector"].astype(str).str.upper() != "FULL"


# -----------------------------
# HELPER FUNCTION
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
plt.figure()
sns.boxplot(data=df, x="is_ga", y="auc")
plt.xticks([0, 1], ["No GA (FULL)", "With GA"])
plt.title("AUC Comparison: GA vs No GA")
save_fig("auc_comparison.png")


# -----------------------------
# 2. F1 COMPARISON
# -----------------------------
plt.figure()
sns.boxplot(data=df, x="is_ga", y="f1")
plt.xticks([0, 1], ["No GA (FULL)", "With GA"])
plt.title("F1 Score Comparison: GA vs No GA")
save_fig("f1_comparison.png")


# -----------------------------
# 3. MODEL COMPARISON
# -----------------------------
plt.figure()
sns.barplot(data=df, x="model", y="auc", hue="is_ga")
plt.title("AUC by Model: GA vs FULL")
save_fig("model_auc.png")


# -----------------------------
# 4. BEST / WORST GA
# -----------------------------
ga_df = df[df["is_ga"]]

best = ga_df.loc[ga_df["auc"].idxmax()]
worst = ga_df.loc[ga_df["auc"].idxmin()]

print("\n🏆 BEST GA RESULT:")
print(best)

print("\n⚠️ WORST GA RESULT:")
print(worst)

# save text summary
with open(FIG_DIR / "best_ga_result.txt", "w") as f:
    f.write("BEST GA RESULT\n\n")
    f.write(str(best))
    f.write("\n\nWORST GA RESULT\n\n")
    f.write(str(worst))

print(f"\n✅ Figures saved in: {FIG_DIR}")