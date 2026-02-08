import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "model/best_model.joblib"
DATA_PATH = "data/day_2011.csv"

# Baseline RMSE from Task 1 (Linear Regression)
# 👉 Replace this value with YOUR actual baseline RMSE
BASELINE_RMSE = 690.80

# Improvement threshold (5% better than baseline)
THRESHOLD_FACTOR = 0.95

# -----------------------------
# Load model
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# -----------------------------
# Load data
# -----------------------------
try:
    data = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

# -----------------------------
# Prepare features & target
# -----------------------------
X = data.drop(columns=["cnt", "dteday"], errors="ignore")
y = data["cnt"]

# -----------------------------
# Predict & evaluate
# -----------------------------
preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))

print(f"Model RMSE: {rmse}")
print(f"Baseline RMSE: {BASELINE_RMSE}")
print(f"Quality Gate Threshold: {THRESHOLD_FACTOR * BASELINE_RMSE}")

# -----------------------------
# Quality Gate (PASS / FAIL)
# -----------------------------
assert rmse <= THRESHOLD_FACTOR * BASELINE_RMSE, (
    "❌ Quality Gate FAILED: Model performance does not meet the threshold."
)

print("✅ Quality Gate PASSED: Model performance is acceptable.")
