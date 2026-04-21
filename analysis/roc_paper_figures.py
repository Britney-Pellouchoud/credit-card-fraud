import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc(model, X_train, X_test, y_train, y_test, name, fig_num):

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title(f"Figure {fig_num}: ROC Curve ({name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # ✅ SAVE HERE
    save_path = f"analysis/figures/roc/figure_{fig_num}_{name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()  # important to avoid memory issues

    return roc_auc