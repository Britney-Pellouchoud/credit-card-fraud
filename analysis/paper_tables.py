import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def evaluate_models(X_train, X_test, y_train, y_test):

    models = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "DT": DecisionTreeClassifier(random_state=42),
        "LR": LogisticRegression(max_iter=1000),
        "NB": GaussianNB(),
        "ANN": MLPClassifier(hidden_layer_sizes=(32,), max_iter=200)
    }

    results = []

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds)
        })

    return pd.DataFrame(results)