import pandas as pd
from pathlib import Path

# GA models
from training_alg.with_ga.train_ann import run_experiment as ann_ga
from training_alg.with_ga.train_dt import run_experiment as dt_ga
from training_alg.with_ga.train_lr import run_experiment as lr_ga
from training_alg.with_ga.train_rf import run_experiment as rf_ga
from training_alg.with_ga.train_nb import run_experiment as nb_ga

# NO GA models
from training_alg.without_ga.train_ann import run_experiment as ann_no
from training_alg.without_ga.train_dt import run_experiment as dt_no
from training_alg.without_ga.train_lr import run_experiment as lr_no
from training_alg.without_ga.train_rf import run_experiment as rf_no
from training_alg.without_ga.train_nb import run_experiment as nb_no


# -----------------------------
# MODEL REGISTRY
# -----------------------------
GA_MODELS = {
    "ann": ann_ga,
    "decision_tree": dt_ga,
    "logistic_regression": lr_ga,
    "random_forest": rf_ga,
    "naive_bayes": nb_ga,
}

NO_GA_MODELS = {
    "ann": ann_no,
    "decision_tree": dt_no,
    "logistic_regression": lr_no,
    "random_forest": rf_no,
    "naive_bayes": nb_no,
}


# -----------------------------
# RUN MODELS
# -----------------------------
def run_group(label, models):
    results = []

    print(f"\n🚀 Running {label} experiments\n")

    for name, fn in models.items():
        print(f"▶ {name}")

        metrics = fn()

        if not metrics:
            print(f"⚠️ {name} returned no metrics")
            continue

        metrics["model"] = name
        metrics["setting"] = label

        results.append(metrics)

    return pd.DataFrame(results)


# -----------------------------
# ANALYSIS
# -----------------------------
def summarize(df):
    return df[["accuracy", "precision", "recall", "f1", "auc"]].describe()


def compare(ga_df, no_ga_df):
    merged = ga_df.merge(no_ga_df, on="model", suffixes=("_ga", "_no_ga"))

    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_ga"] - merged[f"{metric}_no_ga"]

    return merged


# -----------------------------
# MARKDOWN REPORT
# -----------------------------
def write_report(full_df, ga_df, no_ga_df, comparison_df, path="orchestration/results.md"):

    md = []

    md.append("# 📊 GA vs NO-GA Experiment Report\n")

    md.append("## 🧪 Full Model Comparison\n")
    md.append(full_df.sort_values("f1", ascending=False).to_markdown(index=False))
    md.append("\n")

    md.append("## 📈 WITH GA Summary\n")
    md.append(summarize(ga_df).to_markdown())
    md.append("\n")

    md.append("## 📉 WITHOUT GA Summary\n")
    md.append(summarize(no_ga_df).to_markdown())
    md.append("\n")

    md.append("## 🔬 GA Impact (Delta Analysis)\n")
    md.append(comparison_df.sort_values("delta_f1", ascending=False).to_markdown(index=False))
    md.append("\n")

    md.append("## 🏆 Best Models\n")

    md.append("### WITH GA BEST\n")
    md.append(ga_df.sort_values("f1", ascending=False).head(1).to_markdown(index=False))
    md.append("\n")

    md.append("### WITHOUT GA BEST\n")
    md.append(no_ga_df.sort_values("f1", ascending=False).head(1).to_markdown(index=False))

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("\n".join(md))

    print(f"\n💾 Saved markdown report to {path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    ga_df = run_group("WITH_GA", GA_MODELS)
    no_ga_df = run_group("WITHOUT_GA", NO_GA_MODELS)

    full_df = pd.concat([ga_df, no_ga_df], ignore_index=True)

    print("\n📊 FINAL COMPARISON\n")
    print(full_df.sort_values("f1", ascending=False))

    comparison_df = compare(ga_df, no_ga_df)

    write_report(full_df, ga_df, no_ga_df, comparison_df)