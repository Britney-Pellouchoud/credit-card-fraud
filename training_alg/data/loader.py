import pandas as pd


def load_data_smote(path):
    """
    Paper-style loader:
    - loads raw dataset only
    - NO splitting
    - NO SMOTE here (important!)
    """

    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in dataset")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y