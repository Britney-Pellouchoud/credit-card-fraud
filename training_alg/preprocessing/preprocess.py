import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(X):
    """
    Paper step:
    - handle raw numeric features
    - min-max normalization
    """

    X = X.fillna(0)

    print("BEFORE DUMMIES")
    X = X.select_dtypes(include=["number"])
    print("AFTER DUMMIES")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns)