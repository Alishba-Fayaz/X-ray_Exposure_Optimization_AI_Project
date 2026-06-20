#MODULE 1 — Data Pipeline

import os, sys, json, random, csv
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────
BASE      = r"D:\AI_Final_Project"
RAW_DIR   = os.path.join(BASE, "data", "raw")
PROC_DIR  = os.path.join(BASE, "data", "processed")
CAL_FILE  = os.path.join(PROC_DIR, "calibration.json")

IMG_SIZE         = (224, 224)
IMAGES_PER_CLASS = 1000
RANDOM_SEED      = 42
CLASSES          = ["underexposed", "proper", "overexposed"]
CLASS_TO_IDX     = {"underexposed": 0, "proper": 1, "overexposed": 2}

# Retake ground truth (used to build training data for Module 3)
# proper → no retake (0),  under/over → retake (1)
RETAKE_LABEL = {"underexposed": 1, "proper": 0, "overexposed": 1}

# Exposure correction target % (used for Module 4 regression)
# Positive = increase exposure, Negative = decrease
EXPOSURE_CORRECTION = {"underexposed": +25.0, "proper": 0.0, "overexposed": -20.0}


# STEP 1: CALIBRATION — measure raw images

def calibrate():
    paths = list(Path(RAW_DIR).glob("*.jpg")) + \
            list(Path(RAW_DIR).glob("*.png")) + \
            list(Path(RAW_DIR).glob("*.jpeg"))
    if not paths:
        print(f"  ERROR: No images in {RAW_DIR}")
        sys.exit(1)

    random.seed(RANDOM_SEED)
    sample = random.sample(paths, min(80, len(paths)))
    means  = []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None: continue
        gray = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2GRAY)
        means.append(float(gray.mean()))

    raw_mean = float(np.median(means))
    print(f"  Raw image median brightness: {raw_mean:.1f}/255")

    # Compute gamma so underexposed mean ≈ 45, overexposed mean ≈ 210
    # Underexposed (too little radiation) = Bright/White (mean ≈ 210)
    # Overexposed (too much radiation) = Burned/Dark (mean ≈ 45)
    def gamma_for(target):
        r = raw_mean / 255.0
        t = target   / 255.0
        if r <= 0 or r >= 1 or t <= 0 or t >= 1:
            return 1.0
        g_inv = np.log(t) / np.log(r)
        return float(np.clip(1.0 / g_inv, 0.05, 10.0))

    cal = {
        "raw_mean":       raw_mean,
        "gamma_under":    round(gamma_for(180),  3), # Now targets BRIGHT
        "gamma_over":     round(gamma_for(45),   3), # Now targets DARK
        "brighten_under": 15, # Renamed for clarity
        "darken_over":    30, # Renamed for clarity
    }
    os.makedirs(PROC_DIR, exist_ok=True)
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)

    print(f"  gamma_under = {cal['gamma_under']}, gamma_over = {cal['gamma_over']}")
    print(f"  Calibration saved → {CAL_FILE}")
    return cal

# IMAGE TRANSFORMATIONS

def apply_gamma(img, gamma):
    inv = 1.0 / gamma
    lut = np.array([(i/255.0)**inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def apply_brightness(img, delta):
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

def transform(img, cls_name, cal):
    if cls_name == "underexposed":
        # MEDICAL LOGIC: Underexposed gets BRIGHTER
        img = apply_gamma(img, cal["gamma_under"])
        img = apply_brightness(img, cal["brighten_under"]) # Positive value
    elif cls_name == "overexposed":
        # MEDICAL LOGIC: Overexposed gets DARKER
        img = apply_gamma(img, cal["gamma_over"])
        img = apply_brightness(img, -cal["darken_over"])   # Negative value
    return img


# PHYSICS FEATURE EXTRACTION
# Produces the feature rows for CSV files

def extract_features(img_bgr, cls_name):
    """
    Extracts 8 physics-based features from an image.
    These features are used to train Modules 3 and 4.

    Features:
      mean_brightness   — average pixel value (0-255)
      std_deviation     — pixel spread
      entropy           — Shannon information content
      dark_ratio        — fraction of pixels < 51 (20%)
      bright_ratio      — fraction of pixels > 200 (78%)
      contrast_ratio    — max-min range / 255
      median_brightness — median pixel (robust to outliers)
      skewness          — histogram skew (+ve=dark, -ve=bright)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
    mean      = float(gray.mean())
    std       = float(gray.std())
    median    = float(np.median(gray))
    dark_r    = float(np.mean(gray < 51))
    bright_r  = float(np.mean(gray > 200))
    contrast  = float((gray.max() - gray.min()) / 255.0)
    hist, _   = np.histogram(gray / 255.0, bins=32, range=(0,1), density=True)
    h         = hist[hist > 0]
    entropy   = float(-np.sum(h * np.log2(h + 1e-10)) / 5.0)
    mu, sigma = gray.mean(), gray.std()
    skewness  = float(np.mean(((gray - mu) / (sigma + 1e-8)) ** 3))

    return {
        "mean_brightness":   round(mean,     3),
        "std_deviation":     round(std,      3),
        "entropy":           round(entropy,  4),
        "dark_ratio":        round(dark_r,   4),
        "bright_ratio":      round(bright_r, 4),
        "contrast_ratio":    round(contrast, 4),
        "median_brightness": round(median,   3),
        "skewness":          round(skewness, 4),
        "class_label":       cls_name,
        "class_idx":         CLASS_TO_IDX[cls_name],
        "retake":            RETAKE_LABEL[cls_name],
        "exposure_correction": EXPOSURE_CORRECTION[cls_name],
    }

# STEP 4: BUILD DATASET

def build_dataset(cal):
    for split in ["train", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(PROC_DIR, split, cls), exist_ok=True)

    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        paths.extend(list(Path(RAW_DIR).glob(ext)))

    random.seed(RANDOM_SEED)
    random.shuffle(paths)
    paths = paths[:IMAGES_PER_CLASS]

    train_paths, test_paths = train_test_split(
        paths, test_size=0.2, random_state=RANDOM_SEED
    )

    feature_rows = {"train": [], "test": []}
    total_saved  = 0

    for split_name, img_list in [("train", train_paths), ("test", test_paths)]:
        for cls_name in CLASSES:
            save_dir = os.path.join(PROC_DIR, split_name, cls_name)
            saved    = 0

            for idx, img_path in enumerate(img_list):
                raw = cv2.imread(str(img_path))
                if raw is None: continue

                img       = cv2.resize(raw, IMG_SIZE)
                processed = transform(img, cls_name, cal)

                # Quality gate: verify transformation worked
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                mean = gray.mean()
                # NEW LOGIC: Skip if underexposed isn't bright enough, or overexposed isn't dark enough
                if cls_name == "underexposed" and mean < 130: continue 
                if cls_name == "overexposed"  and mean > 100: continue

                # Save image
                out = os.path.join(save_dir, f"{cls_name}_{idx:04d}.jpg")
                cv2.imwrite(out, processed)

                # Extract and store physics features
                feats = extract_features(processed, cls_name)
                feature_rows[split_name].append(feats)
                saved += 1
                total_saved += 1

            print(f"  {split_name:5s}/{cls_name:<15s}: {saved} images")

    # Save feature CSV files
    for split_name, rows in feature_rows.items():
        if not rows: continue
        csv_path = os.path.join(PROC_DIR, f"features_{split_name}.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Features saved → {csv_path}  ({len(rows)} rows)")

    print(f"\n  Total images saved : {total_saved}")


# MAIN
if __name__ == "__main__":
    print("=" * 52)
    print("  MODULE 1 — Data Pipeline")
    print("=" * 52)

    if not os.path.exists(RAW_DIR) or not list(Path(RAW_DIR).glob("*.jpg")):
        os.makedirs(RAW_DIR, exist_ok=True)
        print(f"\n  Place your raw X-ray images in:  {RAW_DIR}")
        print("  Then rerun this script.")
        sys.exit(0)

    cal = calibrate()
    print("\n  Building dataset...")
    build_dataset(cal)
    print("\n  Done. Next: python scripts/train_cnn.py")
