# AI X-Ray Exposure Quality Assessment System

A machine learning pipeline that analyzes chest X-ray images, determines exposure quality, decides if a retake is needed, suggests an exposure correction, and explains its decision visually using Grad-CAM.

---

## Important: X-Ray Physics, Not Photography

In normal photography, overexposed means too bright. In X-ray imaging, it's the opposite, because an X-ray image is a negative.

- **Overexposed (too much radiation)** → image becomes **too dark**, detail is lost. Fix: decrease mAs/kVp.
- **Underexposed (too little radiation)** → image becomes **too bright/washed out**. Fix: increase mAs/kVp.
- **Properly exposed** → balanced mix of dark bone, mid-tone tissue, and bright air.

This logic is used throughout the entire project.

---

## What the System Does

1. Takes an X-ray image as input.
2. Classifies it as **underexposed**, **properly exposed**, or **overexposed**.
3. Decides if the X-ray should be **retaken**.
4. Predicts how much to **adjust exposure** (as a percentage).
5. Generates a **heatmap** showing which part of the image influenced the decision.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add X-ray images to data/raw/

# 3. Build the dataset
python prepare_data.py

# 4. Train the CNN
python train_cnn.py

# 5. Train the decision model
python train_decision_model.py

# 6. Train the regression model
python train_regression_model.py

# 7. Predict from command line
python predict.py --image path/to/xray.jpg

# 8. Or run the web app
python app.py

# open http://localhost:5000
```

---

## Dataset

| Property | Value |
|---|---|
| Source | NIH Chest X-ray Dataset (Kaggle) |
| Raw images | 1000 |
| Labels | Not provided — created using gamma correction |
| Total images after labeling | 3000 (1000 × 3 classes) |
| Train / Test split | 80% / 20% |
| Training images | 2400 (800 per class) |
| Test images | 600 (200 per class) |
| Learning type | Supervised learning |

---

## The Four Models

**1. CNN Classifier** — MobileNetV2, fine-tuned on the X-ray dataset. Looks at the image and classifies exposure type. Test accuracy: **98.5%**.

**2. Decision Model** — Logistic Regression. Decides retake YES/NO using 13 features: 8 physics measurements from the pixels (mean brightness, std deviation, entropy, dark ratio, bright ratio, contrast ratio, median brightness, skewness) plus 5 outputs from the CNN (predicted class, confidence, and the three individual class probabilities). ROC-AUC: **1.000**.

**3. Correction Model** — Ridge Regression. Predicts the exposure correction percentage using the same 8 physics features, expanded with polynomial interaction terms. MAE: **1.93%**, R²: **0.9727**.

**4. Grad-CAM** — Generates a heatmap showing which regions of the X-ray most influenced the CNN's decision. Red/yellow = high influence, blue = low influence.

---

## Why Separate Models Instead of One

- The **CNN** decides what type of exposure error this is.
- The **Decision Model** decides if a retake is needed — using the CNN's own uncertainty (not just its final answer) plus independent pixel statistics.
- The **Regression Model** decides how much to adjust exposure — a continuous question a classifier can't answer.
- **Grad-CAM** explains why the CNN made its decision.

Each one solves a different problem and can be improved independently.

---

## Results Summary

| Model | Metric | Result |
|---|---|---|
| CNN | Test Accuracy | 98.5% |
| Decision Model | ROC-AUC | 1.000 |
| Regression Model | MAE / R² | 1.93% / 0.9727 |

These numbers are high because the dataset was built using gamma correction, which creates very visually distinct classes. Real-world X-rays with subtler exposure issues would likely show lower performance.

---

## Key Fixes Made During Development

1. Made gamma calibration adaptive — values are calculated from the actual input images instead of being hardcoded.
2. Fixed Grad-CAM, which was failing due to how MobileNetV2 is nested inside the model. Solved by tapping into the GlobalAveragePooling2D layer's input.
3. Made sure all recommendation text in the UI matches the correct physics logic.

---

## Limitations

- Exposure labels are simulated, not from real radiographer annotations.
- High accuracy reflects strong separation in this specific dataset, not guaranteed real-world performance.
- Built for learning and demonstration purposes, not for clinical use.

