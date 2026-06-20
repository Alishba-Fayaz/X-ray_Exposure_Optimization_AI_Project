"""
MODULE 6 — The Full Prediction Pipeline

PURPOSE: Single entry point that chains all 4 trained models:
    1. CNN       → exposure class + confidence
    2. Logistic  → retake YES/NO + probability
    3. Regression→ exposure correction %
    4. Grad-CAM  → heatmap showing decision regions
"""

import os, sys, json, argparse
import numpy as np
import cv2
import tensorflow as tf
import joblib

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

# ── Add project root to path ──────────────────────────────────
BASE      = r"D:\AI_Final_Project"
sys.path.insert(0, BASE)

from feature_extraction import extract_physics_features
from explainability      import generate_gradcam

# ── Paths ─────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(BASE, "models")
CNN_PATH     = os.path.join(MODELS_DIR, "cnn_model.keras")
IDX_PATH     = os.path.join(MODELS_DIR, "class_indices.json")
DEC_CLF_PATH = os.path.join(MODELS_DIR, "decision_model.pkl")
DEC_SC_PATH  = os.path.join(MODELS_DIR, "decision_scaler.pkl")
REG_PATH     = os.path.join(MODELS_DIR, "regression_model.pkl")

IMG_SIZE = (224, 224)

CLASS_META = {
    "underexposed": {
        "icon": "🌑", "color": "RED",
        "meaning": "Image too dark — bone detail lost in shadows",
    },
    "proper": {
        "icon": "✅", "color": "GREEN",
        "meaning": "Balanced brightness — suitable for diagnosis",
    },
    "overexposed": {
        "icon": "☀️", "color": "ORANGE",
        "meaning": "Image too bright — bone detail washed out",
    },
}

# MODEL LOADING
def load_all_models():
    missing = []
    for p, name in [(CNN_PATH,     "CNN"),
                    (DEC_CLF_PATH, "Decision model"),
                    (DEC_SC_PATH,  "Decision scaler"),
                    (REG_PATH,     "Regression model"),
                    (IDX_PATH,     "Class indices")]:
        if not os.path.exists(p):
            missing.append(f"  {name}: {p}")

    if missing:
        print("ERROR — Missing model files:")
        print("\n".join(missing))
        print("\nRun the training scripts in order:")
        print("  python scripts/prepare_data.py")
        print("  python scripts/train_cnn.py")
        print("  python scripts/train_decision_model.py")
        print("  python scripts/train_regression_model.py")
        sys.exit(1)

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    cnn     = tf.keras.models.load_model(CNN_PATH)
    dec_clf = joblib.load(DEC_CLF_PATH)
    dec_sc  = joblib.load(DEC_SC_PATH)
    reg     = joblib.load(REG_PATH)

    with open(IDX_PATH) as f:
        name_to_idx = json.load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    return cnn, dec_clf, dec_sc, reg, idx_to_name, preprocess_input


# PREPROCESSING

def load_and_preprocess(image_path, preprocess_input):
    """
    Returns:
      img_bgr    : original image for display (uint8)
      img_batch  : preprocessed for CNN (1,224,224,3) float32
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: Cannot read {image_path}")
        sys.exit(1)

    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_batch   = preprocess_input(np.expand_dims(img_rgb, 0))

    return img_resized, img_batch


# FULL PIPELINE
def run_pipeline(image_path, show_gradcam=True, save_heatmap=None):
    # Load
    (cnn, dec_clf, dec_sc, reg,
     idx_to_name, preprocess_input) = load_all_models()

    img_bgr, img_batch = load_and_preprocess(image_path, preprocess_input)

    # ── STAGE 1: CNN Classification ───────────────────────────
    cnn_probs   = cnn.predict(img_batch, verbose=0)[0]
    cnn_idx     = int(np.argmax(cnn_probs))
    cnn_class   = idx_to_name[cnn_idx]
    cnn_conf    = float(cnn_probs[cnn_idx])

    # ── STAGE 2: Physics Features ─────────────────────────────
    feat_dict, feat_arr = extract_physics_features(img_bgr)

    # ── STAGE 3: Decision Model (Retake YES/NO) ───────────────
    # Build same feature vector used in training
    dec_feats = np.hstack([
        feat_arr,
        [cnn_idx, cnn_conf,
         float(cnn_probs[0]), float(cnn_probs[1]), float(cnn_probs[2])]
    ]).reshape(1, -1)

    dec_feats_sc = dec_sc.transform(dec_feats)
    retake_pred  = int(dec_clf.predict(dec_feats_sc)[0])
    retake_prob  = float(dec_clf.predict_proba(dec_feats_sc)[0][1])

    # ── STAGE 4: Regression (Exposure Correction %) ──────────
    correction_pct = float(reg.predict(feat_arr.reshape(1, -1))[0])

    # ── STAGE 5: Grad-CAM ─────────────────────────────────────
    gradcam_overlay = None
    if show_gradcam or save_heatmap:
        try:
            gradcam_overlay, _ = generate_gradcam(
                cnn, img_batch, cnn_idx
            )
            if save_heatmap:
                cv2.imwrite(save_heatmap, gradcam_overlay)
                print(f"\n  Heatmap saved → {save_heatmap}")
        except Exception as e:
            print(f"  Grad-CAM failed: {e}")

    return {
        "image_path":    image_path,
        "cnn_class":     cnn_class,
        "cnn_conf":      cnn_conf,
        "cnn_probs":     {idx_to_name[i]: float(p)
                          for i, p in enumerate(cnn_probs)},
        "retake":        retake_pred,
        "retake_prob":   retake_prob,
        "correction_pct": correction_pct,
        "physics":       feat_dict,
        "gradcam":       gradcam_overlay,
    }

# DISPLAY
def print_results(r):
    cls  = r["cnn_class"]
    meta = CLASS_META.get(cls, {})

    print("\n" + "═"*56)
    print("   AI X-RAY EXPOSURE ASSESSMENT — FULL PIPELINE")
    print("═"*56)
    print(f"  Image : {os.path.basename(r['image_path'])}")

    print(f"\n  ── STAGE 1: CNN Classification ──")
    print(f"  Result     : {meta.get('icon','')}  {cls.upper()}")
    print(f"  Confidence : {r['cnn_conf']*100:.1f}%")
    print(f"  Meaning    : {meta.get('meaning','')}")
    print()
    print("  Class probability breakdown:")
    for c, p in sorted(r["cnn_probs"].items(), key=lambda x: x[1], reverse=True):
        bar  = "█" * int(p*24) + "░" * (24 - int(p*24))
        icon = CLASS_META.get(c, {}).get("icon", "")
        print(f"    {icon} {c:<16} [{bar}] {p*100:.1f}%")

    print(f"\n  ── STAGE 2: Physics Measurements ──")
    p = r["physics"]
    print(f"  Mean brightness   : {p['mean_brightness']:.1f}/255  "
          f"({'dark' if p['mean_brightness']<80 else 'bright' if p['mean_brightness']>170 else 'balanced'})")
    print(f"  Std deviation     : {p['std_deviation']:.1f}  "
          f"({'low=flat' if p['std_deviation']<30 else 'normal'})")
    print(f"  Entropy           : {p['entropy']:.3f}  "
          f"({'low=lost detail' if p['entropy']<0.4 else 'normal'})")
    print(f"  Dark pixel ratio  : {p['dark_ratio']*100:.1f}%")
    print(f"  Bright pixel ratio: {p['bright_ratio']*100:.1f}%")

    print(f"\n  ── STAGE 3: Retake Decision (Logistic Regression) ──")
    retake_str = "YES — Retake required ⚠️" if r["retake"] else "NO — Image acceptable ✅"
    print(f"  Decision      : {retake_str}")
    print(f"  Retake prob   : {r['retake_prob']*100:.1f}%")
    print(f"  Explanation   : Model trained on 8 physics + 5 CNN features")

    print(f"\n  ── STAGE 4: Exposure Correction (Ridge Regression) ──")
    corr = r["correction_pct"]
    direction = "INCREASE" if corr > 0 else "DECREASE" if corr < 0 else "NO CHANGE"
    print(f"  Adjustment    : {corr:+.1f}%  →  {direction} exposure")
    if   corr > 0:  print(f"  Action        : Increase mAs or kVp by ~{abs(corr):.0f}%")
    elif corr < 0:  print(f"  Action        : Decrease mAs or kVp by ~{abs(corr):.0f}%")
    else:           print(f"  Action        : No adjustment needed")

    if r["gradcam"] is not None:
        print(f"\n  ── STAGE 5: Grad-CAM ──")
        print(f"  Heatmap generated. Red=high influence, Blue=low influence.")

    print("\n" + "═"*56)

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI X-Ray Exposure Assessment — Full Pipeline"
    )
    parser.add_argument("--image",        required=True,  help="Path to X-ray image")
    parser.add_argument("--no-gradcam",   action="store_true", help="Skip Grad-CAM")
    parser.add_argument("--save-heatmap", default=None,   help="Save heatmap to path")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"  Image not found: {args.image}")
        sys.exit(1)

    result = run_pipeline(
        args.image,
        show_gradcam=not args.no_gradcam,
        save_heatmap=args.save_heatmap,
    )
    print_results(result)

    # Show Grad-CAM window if display available
    if result["gradcam"] is not None and not args.save_heatmap:
        try:
            cv2.imshow("Grad-CAM Heatmap", result["gradcam"])
            print("  Press any key to close heatmap window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            print("  (Display not available — use --save-heatmap to save)")
