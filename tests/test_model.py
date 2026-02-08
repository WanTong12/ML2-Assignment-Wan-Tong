import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

MODEL_PATH = "model/Improved_model.joblib"
DATA_PATH = "data/day_2011.csv"

# Baseline RMSE from Task 1 (Linear Regression) - replace with your actual value
BASELINE_RMSE = 690.80

# Improvement threshold (5% better than baseline)
THRESHOLD_FACTOR = 0.95


def main():
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
    # Feature engineering (MUST match training)
    # -----------------------------
    if "dteday" in data.columns:
        data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce", dayfirst=True)

        # Always create the exact features required by your model
        data["month"] = data["dteday"].dt.month
        data["dayofweek"] = data["dteday"].dt.dayofweek
        data["year"] = data["dteday"].dt.year

    # -----------------------------
    # Prepare features & target
    # -----------------------------
    if "cnt" not in data.columns:
        print("❌ Target column 'cnt' not found in dataset.")
        sys.exit(1)

    y = data["cnt"]

    # Drop target + raw date
    X = data.drop(columns=["cnt", "dteday"], errors="ignore")

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
            print(f"⚠️ Dropping extra features not used in training: {extra}")
            X = X.drop(columns=extra)

        # Reorder columns to match training
        X = X[expected]
    else:
        print("⚠️ Model has no 'feature_names_in_' attribute.")
        print("   Ensure X columns and order match the training pipeline.")
        # Continue anyway, but may fail if order mismatches

    # -----------------------------
    # Predict & evaluate
    # -----------------------------
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))

    threshold = THRESHOLD_FACTOR * BASELINE_RMSE

    print(f"Model RMSE: {rmse:.4f}")
    print(f"Baseline RMSE: {BASELINE_RMSE:.4f}")
    print(f"Quality Gate Threshold (<=): {threshold:.4f}")

    # -----------------------------
    # Quality Gate (PASS / FAIL)
    # -----------------------------
    assert rmse <= threshold, (
        "❌ Quality Gate FAILED: Model performance does not meet the threshold."
    )

    print("✅ Quality Gate PASSED: Model performance is acceptable.")


if __name__ == "__main__":
    main()
