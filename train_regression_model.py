"""
MODULE 4 — Exposure Adjustment Model (Linear Regression)
PURPOSE:
  Predict the exact exposure correction percentage needed:
    e.g: "+18.5% increase" or "-12.3% decrease"

  This is NOT a hardcoded lookup table.
  It is a trained regression model that learns the
  relationship between image brightness features
  and required exposure adjustment.

WHY LINEAR REGRESSION:
  - Interpretable: each feature has a coefficient
  - Appropriate for a continuous numeric target
  - Easy to explain: "output = weighted sum of features + bias"
  - Ridge regularisation prevents overfitting

INPUT FEATURES (same 8 physics features):
  mean_brightness, std_deviation, entropy,
  dark_ratio, bright_ratio, contrast_ratio,
  median_brightness, skewness

TARGET:
  exposure_correction (%)
    underexposed → +25.0
    proper       →   0.0
    overexposed  → -20.0

  These are starting targets. The model learns continuous relationships, so intermediate images
  get intermediate correction values — not just the three fixed numbers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

BASE      = r"D:\AI_Final_Project"
PROC_DIR   = os.path.join(BASE, "data",   "processed")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PHYS_COLS = [
    "mean_brightness", "std_deviation", "entropy",
    "dark_ratio", "bright_ratio", "contrast_ratio",
    "median_brightness", "skewness"
]


def load_data(split):
    csv_path = os.path.join(PROC_DIR, f"features_{split}.csv")
    df = pd.read_csv(csv_path)
    X  = df[PHYS_COLS].values.astype(np.float32)
    y  = df["exposure_correction"].values.astype(np.float32)
    return X, y, df


def train_regression(X_train, y_train):
    """
    Pipeline:
      1. PolynomialFeatures(degree=2) — interaction terms
      2. StandardScaler              — normalise features
      3. Ridge regression            — L2-regularised linear model

    Ridge prevents the polynomial terms from overfitting.
    alpha=1.0 is the regularisation strength.
    """
    pipeline = Pipeline([
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_regression(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print(f"  MAE (Mean Absolute Error) : {mae:.2f}%")
    print(f"  R² Score                  : {r2:.4f}")
    print(f"  Interpretation: on average, prediction is off by {mae:.2f}%")

    # Save report
    report_path = os.path.join(MODELS_DIR, "regression_report.txt")
    with open(report_path, "w") as f:
        f.write("EXPOSURE ADJUSTMENT MODEL — RIDGE REGRESSION\n")
        f.write("=" * 42 + "\n")
        f.write(f"MAE  : {mae:.4f}%\n")
        f.write(f"R²   : {r2:.4f}\n")
        sample_preds = list(zip(y_test[:10], y_pred[:10]))
        f.write("\nSample predictions (actual vs predicted):\n")
        for act, pred in sample_preds:
            f.write(f"  Actual: {act:+.1f}%   Predicted: {pred:+.2f}%\n")
    print(f"  Report saved → {report_path}")

    # Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#00d4ff", edgecolors="none", s=20)
    lims = [min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "--", color="#ff5252", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Correction (%)")
    ax.set_ylabel("Predicted Correction (%)")
    ax.set_title(f"Regression: Actual vs Predicted\nMAE={mae:.2f}%  R²={r2:.4f}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "regression_actual_vs_predicted.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Scatter plot → {out}")

    return mae, r2


if __name__ == "__main__":
    print("=" * 52)
    print("  MODULE 4 — Exposure Adjustment Regression")
    print("=" * 52)

    X_train, y_train, _ = load_data("train")
    X_test,  y_test,  _ = load_data("test")

    print(f"\n  Train: {X_train.shape[0]} samples")
    print(f"  Test : {X_test.shape[0]} samples")
    print(f"  Target range: {y_train.min():.1f}% to {y_train.max():.1f}%")

    pipeline = train_regression(X_train, y_train)
    print("\n  Evaluation on test set:")
    mae, r2 = evaluate_regression(pipeline, X_test, y_test)

    joblib.dump(pipeline, os.path.join(MODELS_DIR, "regression_model.pkl"))
    print(f"\n  Saved → regression_model.pkl")
    print("  Next: python scripts/predict.py --image <path>")
