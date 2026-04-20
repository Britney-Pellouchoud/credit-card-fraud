import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data(path):
    df = pd.read_csv(path)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


if __name__ == "__main__":

    X, y = load_data("../train_processed.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(max_depth=10, random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n🌳 Decision Tree (NO GA)")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))