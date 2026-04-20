import os
import pandas as pd

import os
import pandas as pd

# Get the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to data folder (sibling of random_scripts)
data_folder = os.path.join(script_dir, "../data")
original_csv = "/Users/britneypellouchoud/credit-card-fraud/data/creditcard_2023.csv"
sample_csv = (
    "/Users/britneypellouchoud/credit-card-fraud/data/creditcard_2023_sample100.csv"
)

# Check if the original CSV exists
if not os.path.exists(original_csv):
    raise FileNotFoundError(f"Original CSV not found: {original_csv}")

# Load full dataset
df = pd.read_csv(original_csv)

# Take a random sample of 100 rows
sample_df = df.sample(
    n=100, random_state=None
)  # random_state=None gives a fresh random sample each time

# Save the sample (overwrites if it exists)
sample_df.to_csv(sample_csv, index=False)

print(f"Fresh 100-row sample created/overwritten: {sample_csv}")

# Paths
data_folder = "../data"

# Check if the original CSV exists
if not os.path.exists(original_csv):
    raise FileNotFoundError(f"Original CSV not found: {original_csv}")

# Load full dataset
df = pd.read_csv(original_csv)

# Take a random sample of 100 rows
sample_df = df.sample(
    n=100, random_state=None
)  # random_state=None gives a fresh random sample each time

# Save the sample (overwrites if it exists)
sample_df.to_csv(sample_csv, index=False)

print(f"Fresh 100-row sample created/overwritten: {sample_csv}")
