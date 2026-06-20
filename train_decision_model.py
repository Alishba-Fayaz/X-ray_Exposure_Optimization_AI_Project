"""
MODULE 3 — Retake Decision Model (Logistic Regression)
PURPOSE:
  Predict whether an X-ray should be RETAKEN (YES/NO).
  This is NOT a rule-based if-else system.
  It is a trained binary classifier on real features.

WHY LOGISTIC REGRESSION:
  - Interpretable: coefficients show which features matter most
  - Fast to train and predict on CPU
  - Works well when features are informative (ours are)
  - Output is a probability: P(retake=YES) — clinically meaningful
  - Easy to explain in viva: "it learned the decision boundary
    from data, not from hardcoded rules"

TOTAL: 11 features

OUTPUT:
  0 = No retake needed
  1 = Retake required
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import seaborn as sns

BASE      = r"D:\AI_Final_Project"
PROC_DIR   = os.path.join(BASE, "data",   "processed")
MODELS_DIR = os.path.join(BASE, "models")
CNN_MODEL  = os.path.join(MODELS_DIR, "cnn_model.keras")
IDX_FILE   = os.path.join(MODELS_DIR, "class_indices.json")

os.makedirs(MODELS_DIR, exist_ok=True)


# FEATURE BUILDER
# Combines CSV physics features with CNN predictions

def load_cnn_and_indices():
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = tf.keras.models.load_model(CNN_MODEL)
    with open(IDX_FILE) as f:
        name_to_idx = json.load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    return model, idx_to_name, preprocess_input


def get_cnn_predictions_for_dataset(split, cnn_model, preprocess_input):
    """
    Run CNN inference on all images in data/processed/{split}/
    and return array of (class_idx, confidence, feature_vector).
    """
    import cv2
    IMG_SIZE   = (224, 224)
    split_dir  = os.path.join(PROC_DIR, split)
    CLASSES    = ["underexposed", "proper", "overexposed"]

    rows = []
    for cls_name in CLASSES:
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cls_dir): continue
        imgs = [f for f in os.listdir(cls_dir)
                if f.endswith((".jpg", ".png"))]

        for fname in imgs:
            path = os.path.join(cls_dir, fname)
            img  = cv2.imread(path)
            if img is None: continue
            img  = cv2.resize(img, IMG_SIZE).astype(np.float32)
            inp  = preprocess_input(np.expand_dims(img, 0))
            prob = cnn_model.predict(inp, verbose=0)[0]
            rows.append({
                "cnn_pred":  int(np.argmax(prob)),
                "cnn_conf":  float(prob.max()),
                "cnn_prob_0": float(prob[0]),
                "cnn_prob_1": float(prob[1]),
                "cnn_prob_2": float(prob[2]),
                "fname":     fname,
                "cls_name":  cls_name,
            })
    return pd.DataFrame(rows)


def build_feature_matrix(split, cnn_model=None, preprocess_input=None):
    """
    Merge physics CSV + CNN predictions → feature matrix X, labels y.
    """
    csv_path = os.path.join(PROC_DIR, f"features_{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run prepare_data.py first.")

    phys_df = pd.read_csv(csv_path)

    # Physics features
    phys_cols = [
        "mean_brightness", "std_deviation", "entropy",
        "dark_ratio", "bright_ratio", "contrast_ratio",
        "median_brightness", "skewness"
    ]

    if cnn_model is not None:
        print(f"  Running CNN on {split} images for feature extraction...")
        cnn_df = get_cnn_predictions_for_dataset(split, cnn_model, preprocess_input)
        
        # Since both follow same ordering, we use class_idx from CSV
        # and cnn predictions ordered same way
        phys_df = phys_df.reset_index(drop=True)
        cnn_df  = cnn_df.reset_index(drop=True)

        # If lengths differ, truncate to smaller
        n = min(len(phys_df), len(cnn_df))
        phys_df = phys_df.iloc[:n]
        cnn_df  = cnn_df.iloc[:n]

        X = np.hstack([
            phys_df[phys_cols].values,
            cnn_df[["cnn_pred", "cnn_conf",
                    "cnn_prob_0", "cnn_prob_1", "cnn_prob_2"]].values,
        ])
    else:
        X = phys_df[phys_cols].values

    y_retake = phys_df["retake"].values
    return X, y_retake, phys_df


# TRAINING

def train_decision_model(X_train, y_train):
    """
    Logistic Regression with L2 regularisation.

    C=1.0 : inverse regularisation strength
             smaller C = more regularisation
    solver='lbfgs' : good for small-medium datasets
    max_iter=1000  : ensure convergence
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",  # handles any class imbalance
    )
    clf.fit(X_sc, y_train)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_sc, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return clf, scaler


def evaluate_decision_model(clf, scaler, X_test, y_test, feature_names):
    X_sc   = scaler.transform(X_test)
    y_pred = clf.predict(X_sc)
    y_prob = clf.predict_proba(X_sc)[:, 1]

    report = classification_report(y_test, y_pred,
                                    target_names=["No Retake", "Retake"])
    print("\n  Classification Report:\n", report)

    # ROC-AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC: {auc:.4f}")

    # Save report
    report_path = os.path.join(MODELS_DIR, "decision_report.txt")
    with open(report_path, "w") as f:
        f.write("DECISION MODEL — LOGISTIC REGRESSION\n")
        f.write("=" * 40 + "\n")
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")
    print(f"  Report saved → {report_path}")

    # Feature importance (coefficients)
    coef = clf.coef_[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#ff5252" if c < 0 else "#4caf50" for c in coef]
    ax.barh(feature_names, coef, color=colors)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_title("Logistic Regression Feature Coefficients\n"
                 "(positive = increases retake probability)")
    ax.set_xlabel("Coefficient value")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "decision_coefficients.png")
    plt.savefig(out, dpi=150, facecolor="#1a1a2e"); plt.close()
    print(f"  Coefficients chart → {out}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#00d4ff", label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Retake Decision")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "decision_roc.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  ROC curve → {out}")


if __name__ == "__main__":
    print("=" * 52)
    print("  MODULE 3 — Retake Decision Model")
    print("=" * 52)

    # Try to use CNN features; fall back to physics-only if model missing
    if os.path.exists(CNN_MODEL):
        cnn_model, _, preprocess_input = load_cnn_and_indices()
        print("  CNN loaded — using CNN + physics features")
    else:
        cnn_model = None
        preprocess_input = None
        print("  CNN not found — using physics features only")

    print("\n  Loading training features...")
    X_train, y_train, phys_train = build_feature_matrix(
        "train", cnn_model, preprocess_input
    )

    print("  Loading test features...")
    X_test, y_test, phys_test = build_feature_matrix(
        "test", cnn_model, preprocess_input
    )

    phys_cols = [
        "mean_brightness", "std_deviation", "entropy",
        "dark_ratio", "bright_ratio", "contrast_ratio",
        "median_brightness", "skewness"
    ]
    feature_names = phys_cols + (
        ["cnn_pred", "cnn_conf", "cnn_prob_0", "cnn_prob_1", "cnn_prob_2"]
        if cnn_model else []
    )

    print(f"\n  Training on {X_train.shape[0]} samples, "
          f"{X_train.shape[1]} features")
    print(f"  Class balance: {np.bincount(y_train)}")

    clf, scaler = train_decision_model(X_train, y_train)
    evaluate_decision_model(clf, scaler, X_test, y_test, feature_names)

    # Save
    joblib.dump(clf,    os.path.join(MODELS_DIR, "decision_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "decision_scaler.pkl"))
    print(f"\n  Saved: decision_model.pkl + decision_scaler.pkl")
    print("  Next: python scripts/train_regression_model.py")
