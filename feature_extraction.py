"""
Shared physics feature extraction used by all modules, Single source of truth.
"""

import numpy as np
import cv2

def extract_physics_features(img_bgr):
    """
    Extract 8 physics-based features from a BGR image.
    Returns a dict and a numpy array (same features, both forms).
    These features are the physical measurement of exposure:
      mean_brightness   — average pixel intensity (0-255)
      std_deviation     — tonal spread
      entropy           — Shannon information content
      dark_ratio        — fraction of pixels below 51 (<20% max)
      bright_ratio      — fraction of pixels above 200 (>78% max)
      contrast_ratio    — (max-min) / 255
      median_brightness — median pixel (robust to outliers)
      skewness          — histogram asymmetry
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)

    mean    = float(gray.mean())
    std     = float(gray.std())
    median  = float(np.median(gray))
    dark_r  = float(np.mean(gray < 51))
    bright_r= float(np.mean(gray > 200))
    contrast= float((gray.max() - gray.min()) / 255.0)

    hist, _ = np.histogram(gray / 255.0, bins=32, range=(0, 1), density=True)
    h       = hist[hist > 0]
    entropy = float(-np.sum(h * np.log2(h + 1e-10)) / 5.0)

    mu, sigma = gray.mean(), gray.std()
    skewness  = float(np.mean(((gray - mu) / (sigma + 1e-8)) ** 3))

    feat_dict = {
        "mean_brightness":    round(mean,     3),
        "std_deviation":      round(std,      3),
        "entropy":            round(entropy,  4),
        "dark_ratio":         round(dark_r,   4),
        "bright_ratio":       round(bright_r, 4),
        "contrast_ratio":     round(contrast, 4),
        "median_brightness":  round(median,   3),
        "skewness":           round(skewness, 4),
    }

    feat_arr = np.array([
        mean, std, entropy, dark_r, bright_r,
        contrast, median, skewness
    ], dtype=np.float32)

    return feat_dict, feat_arr

FEATURE_NAMES = [
    "mean_brightness", "std_deviation", "entropy",
    "dark_ratio", "bright_ratio", "contrast_ratio",
    "median_brightness", "skewness"
]
