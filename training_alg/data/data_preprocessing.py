import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42, use_smote=True):
        self.test_size = test_size
        self.random_state = random_state
        self.use_smote = use_smote
        self.scaler = MinMaxScaler()

    def load_data(self, path):
        df = pd.read_csv(path)
        print(f"Loaded data: {df.shape}")
        return df

    def detect_target_column(self, df):
        possible_targets = ["Class", "class", "is_fraud", "Is Fraud"]

        for col in possible_targets:
            if col in df.columns:
                return col

        raise ValueError("No valid target column found.")

    def split_X_y(self, df, target_col=None):
        if target_col is None:
            target_col = self.detect_target_column(df)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        return X, y

    def split_train_test(self, X, y):
        return train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

    def scale(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        return X_train_scaled, X_test_scaled

    def apply_smote(self, X_train, y_train):
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print("\nAfter SMOTE:")
        print(y_resampled.value_counts())

        return X_resampled, y_resampled

    def save_data(self, X_train, X_test, y_train, y_test):
        # Combine features + target
        train_df = X_train.copy()
        train_df["target"] = y_train.values

        test_df = X_test.copy()
        test_df["target"] = y_test.values

        train_df.to_csv("train_processed.csv", index=False)
        test_df.to_csv("test_processed.csv", index=False)

        print("\nSaved:")
        print("train_processed.csv")
        print("test_processed.csv")

    def run(self, filepath):
        df = self.load_data(filepath)

        X, y = self.split_X_y(df)

        X_train, X_test, y_train, y_test = self.split_train_test(X, y)

        print("\nBefore SMOTE:")
        print(y_train.value_counts())

        X_train, X_test = self.scale(X_train, X_test)

        if self.use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)

        self.save_data(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test


# run full dataset
if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    X_train, X_test, y_train, y_test = preprocessor.run("creditcard_2023.csv")

    print("\nFinal Shapes:")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
