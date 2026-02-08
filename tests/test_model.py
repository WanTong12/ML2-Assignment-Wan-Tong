import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

MODEL_PATH = "model/Improved_model.joblib"
DATA_PATH = "data/day_2011.csv"

# Replace with YOUR baseline RMSE from Task 1 (Linear Regression)
BASELINE_RMSE = 690.80
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
# Feature engineering (ONLY if training used these)
# -----------------------------
if "dteday" in data.columns:
    data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce", dayfirst=True)

    # Create engineered features only if they are NOT already present
    if "month" not in data.columns and "mnth" not in data.columns:
        data["month"] = data["dteday"].dt.month

    if "dayofweek" not in data.columns and "weekday" not in data.columns:
        data["dayofweek"] = data["dteday"].dt.dayofweek

    if "year" not in data.columns and "yr" not in data.columns:
        data["year"] = data["dteday"].dt.year

# -----------------------------
# Prepare X, y
# -----------------------------
if "cnt" not in data.columns:
    print("❌ Target column 'cnt' not found in dataset.")
    sys.exit(1)

y = data["cnt"]

# Drop target + raw date
drop_cols = ["cnt"]
if "dteday" in data.columns:
    drop_cols.append("dteday")

X = data.drop(columns=drop_cols, errors="ignore")

# -----------------------------
# Align features to training schema
# -----------------------------
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)

    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]

    if missing:
        print(f"❌ Missing features required by model: {missing}")
        sys.exit(1)

    if extra:
        # Not always fatal, but it's safer to drop extras so the gate is consistent
        print(f"⚠️ Dropping extra features not used in training: {extra}")
        X = X.drop(columns=extra)

    # Reorder to match training
    X = X[expected]
else:
    print("⚠️ Model has no 'feature_names_in_' attribute. Ensure X column order matches training manually.")

# -----------------------------
# Predict & evaluate
# -----------------------------
preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))

threshold = THRESHOLD_FACTOR * BASELINE_RMSE

print(f"Model RMSE: {rmse:.4f}")
print(f"Baseline RMSE: {BASELINE_RMSE:.4f}")
print(f"Quality Gate Threshold (<=): {threshold:.4f}")

assert rmse <= threshold, "❌ Quality Gate FAILED: Model performance does not meet the threshold."

print("✅ Quality Gate PASSED: Model performance is acceptable.")
