import numpy as np
import random

# --------------------------
# MODE SWITCH
# --------------------------
DEBUG = False   # ⚠ must be Python boolean, not FALSE

PAPER_MODE = not DEBUG   # clean global switch

# --------------------------
# OUTPUT DIRECTORIES
# --------------------------
FIGURE_DIR = "analysis/figures"
ROC_DIR = f"{FIGURE_DIR}/roc"
CONVERGENCE_DIR = f"{FIGURE_DIR}/convergence"

# --------------------------
# REPRODUCIBILITY SEED (IMPORTANT)
# --------------------------
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# --------------------------
# DATA / GA SETTINGS
# --------------------------
if DEBUG:
    # FAST MODE (development only)
    SAMPLE_SIZE = 20000
    GA_POP_SIZE = 5
    GA_GENERATIONS = 3
    RF_ESTIMATORS_GA = 10
    RF_ESTIMATORS_FINAL = 20
    RUN_ONLY_ONE_VECTOR = True
else:
    # FULL PAPER MODE
    SAMPLE_SIZE = None
    GA_POP_SIZE = 20
    GA_GENERATIONS = 30
    RF_ESTIMATORS_GA = 50
    RF_ESTIMATORS_FINAL = 100
    RUN_ONLY_ONE_VECTOR = False