import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "training_alg", "data", "processed", "train.csv")

FEATURE_PATH = os.path.join(
    BASE_DIR, "training_alg", "with_ga", "selected_features.json"
)
