"""
Flask Web Server
Open: http://localhost:5000
"""

import os, sys, json, base64
import numpy as np
import cv2
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify, send_from_directory

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

BASE = r"D:\AI_Final_Project"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

MODELS_DIR = os.path.join(BASE, "models")
IMG_SIZE   = (224, 224)
ALLOWED    = {"jpg", "jpeg", "png"}

CLASS_CFG = {
    "underexposed": {
        "label": "Underexposed",
        "rec":   "Image is too bright. Increase mAs or kVp and retake.",
    },
    "proper": {
        "label": "Properly Exposed",
        "rec":   "Exposure is correct. Suitable for diagnostic review.",
    },
    "overexposed": {
        "label": "Overexposed",
        "rec":   "Image is too dark. Decrease mAs or kVp and retake.",
    },
}

# PHYSICS FEATURE EXTRACTION 

def extract_physics_features(img_bgr):
    """
    Extract 8 physics-based features directly from the BGR image.
    Returns (dict for display, numpy array for ML models).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)

    mean     = float(gray.mean())
    std      = float(gray.std())
    median   = float(np.median(gray))
    dark_r   = float(np.mean(gray < 51))
    bright_r = float(np.mean(gray > 200))
    contrast = float((gray.max() - gray.min()) / 255.0)

    hist, _  = np.histogram(gray / 255.0, bins=32, range=(0, 1), density=True)
    h        = hist[hist > 0]
    entropy  = float(-np.sum(h * np.log2(h + 1e-10)) / 5.0)

    mu       = gray.mean()
    sigma    = gray.std()
    skewness = float(np.mean(((gray - mu) / (sigma + 1e-8)) ** 3))

    feat_dict = {
        "Mean Brightness":   f"{mean:.1f} / 255",
        "Std Deviation":     f"{std:.1f}",
        "Entropy":           f"{entropy:.3f}",
        "Dark Pixel Ratio":  f"{dark_r*100:.1f}%",
        "Bright Pixel Ratio":f"{bright_r*100:.1f}%",
        "Contrast Ratio":    f"{contrast:.3f}",
        "Median Brightness": f"{median:.1f}",
        "Skewness":          f"{skewness:.3f}",
    }

    feat_arr = np.array(
        [mean, std, entropy, dark_r, bright_r, contrast, median, skewness],
        dtype=np.float32
    )

    return feat_dict, feat_arr


# GRAD-CAM 

def generate_gradcam(model, img_batch, class_index, alpha=0.5):
    """
    Compute Grad-CAM heatmap by targeting the GAP layer's input.
    This bypasses the MobileNetV2 black-box disconnection issue perfectly.
    """
    # ── Step 1: Find the GAP layer by TYPE 
    gap_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
            break
            
    if gap_layer is None:
        raise RuntimeError("Could not find a GlobalAveragePooling2D layer in the model.")

    # ── Step 2: Build gradient model using GAP input 
    # gap_layer.input is the spatial map coming out of MobileNetV2
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[gap_layer.input, model.output]
    )

    # ── Step 3: Forward pass + gradient computation 
    img_tensor = tf.cast(img_batch, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        # Explicitly watch the intermediate spatial tensor
        tape.watch(conv_outputs) 
        class_score = predictions[:, class_index]

    # Gradient of class score w.r.t. conv feature maps
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradients returned None. Ensure tape.watch() is applied correctly.")

    # ── Step 4: Pool → weight → heatmap ──────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_outputs[0]                              # (H, W, C)
    heatmap  = conv_out @ pooled_grads[..., tf.newaxis]     # (H, W, 1)
    heatmap  = tf.squeeze(heatmap).numpy()                  # (H, W)

    # ReLU: keep only positive activations
    heatmap  = np.maximum(heatmap, 0)

    # Normalise to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # ── Step 5: Resize and colorize ───────────────────────────
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert original image from MobileNetV2 range [-1,1] → [0,255]
    orig_np      = img_batch[0]
    if hasattr(orig_np, "numpy"):
        orig_np = orig_np.numpy()
    orig_display = np.clip((orig_np + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    orig_bgr     = cv2.cvtColor(orig_display, cv2.COLOR_RGB2BGR)

    # Blend: (1-alpha)*original + alpha*heatmap
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_color, alpha, 0)

    return overlay, heatmap_resized

def to_base64_png(img_bgr):
    """Encode a BGR image to base64 PNG string for embedding in JSON."""
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


# LOAD ALL MODELS AT STARTUP
print("\n" + "=" * 52)
print("  X-Ray Exposure AI  —  Starting Server")
print("=" * 52)

cnn = dec_clf = dec_sc = reg = None
idx_to_name      = {}
preprocess_input = None

try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    cnn = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_model.keras"))
    print(f"  CNN loaded  ({len(cnn.layers)} layers)")
    # Print all layer names so we know what Grad-CAM will find
    conv_layers = [l.name for l in cnn.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        # Check inside sub-models
        for l in cnn.layers:
            if hasattr(l, "layers"):
                conv_layers += [sl.name for sl in l.layers
                                if isinstance(sl, tf.keras.layers.Conv2D)]
    print(f"  Conv layers found: {len(conv_layers)} — last: {conv_layers[-1] if conv_layers else 'NONE'}")
except Exception as e:
    print(f"  CNN FAILED: {e}")

try:
    with open(os.path.join(MODELS_DIR, "class_indices.json")) as f:
        name_to_idx = json.load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    print(f"  Class map: {idx_to_name}")
except Exception as e:
    print(f"  Class indices FAILED: {e}")

try:
    dec_clf = joblib.load(os.path.join(MODELS_DIR, "decision_model.pkl"))
    dec_sc  = joblib.load(os.path.join(MODELS_DIR, "decision_scaler.pkl"))
    print("  Decision model loaded")
except Exception as e:
    print(f"  Decision model FAILED: {e}")

try:
    reg = joblib.load(os.path.join(MODELS_DIR, "regression_model.pkl"))
    print("  Regression model loaded")
except Exception as e:
    print(f"  Regression model FAILED: {e}")

print("=" * 52 + "\n")


#helpers

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def decode_image(file_bytes):
    """Decode uploaded bytes → BGR image + preprocessed CNN batch."""
    arr       = np.frombuffer(file_bytes, np.uint8)
    img       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img_bgr   = cv2.resize(img, IMG_SIZE)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_batch = preprocess_input(np.expand_dims(img_rgb, 0))
    return img_bgr, img_batch


#routes
@app.route("/")
def index():
    # Serve index.html from the same folder as app.py
    here = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(here, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(here, filename)


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validation
    if cnn is None or not idx_to_name:
        return jsonify({"error": "CNN not loaded. Check terminal for details."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Please upload a JPG or PNG file."}), 400

    file_bytes       = file.read()
    img_bgr, img_batch = decode_image(file_bytes)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image. File may be corrupt."}), 400

    # ── Stage 1: CNN Classification 
    cnn_probs = cnn.predict(img_batch, verbose=0)[0]
    cnn_idx   = int(np.argmax(cnn_probs))
    cnn_class = idx_to_name[cnn_idx]
    cnn_conf  = float(cnn_probs[cnn_idx])

    # ── Stage 2: Physics Features
    feat_dict, feat_arr = extract_physics_features(img_bgr)

    # ── Stage 3: Retake Decision
    retake_pred, retake_prob = 0, 0.0
    if dec_clf is not None and dec_sc is not None:
        try:
            dec_feats    = np.hstack([
                feat_arr,
                [cnn_idx, cnn_conf,
                 float(cnn_probs[0]), float(cnn_probs[1]), float(cnn_probs[2])]
            ]).reshape(1, -1)
            dec_sc_feats = dec_sc.transform(dec_feats)
            retake_pred  = int(dec_clf.predict(dec_sc_feats)[0])
            retake_prob  = float(dec_clf.predict_proba(dec_sc_feats)[0][1])
        except Exception as e:
            print(f"  Decision model error: {e}")

    # ── Stage 4: Correction % 
    correction_pct = 0.0
    if reg is not None:
        try:
            correction_pct = float(reg.predict(feat_arr.reshape(1, -1))[0])
        except Exception as e:
            print(f"  Regression error: {e}")

    # ── Stage 5: Grad-CAM 
    gradcam_b64   = None
    gradcam_error = None
    try:
        overlay, _  = generate_gradcam(cnn, img_batch, cnn_idx, alpha=0.5)
        gradcam_b64 = to_base64_png(overlay)
        print("  Grad-CAM generated successfully")
    except Exception as e:
        gradcam_error = str(e)
        print(f"  Grad-CAM FAILED: {e}")

    # ── Build response 
    cfg       = CLASS_CFG.get(cnn_class, CLASS_CFG["proper"])
    corr      = round(correction_pct, 1)
    direction = "Increase" if corr > 0 else "Decrease" if corr < 0 else "No change"

    # Original image base64 (for display in UI)
    orig_b64 = base64.b64encode(file_bytes).decode()

    return jsonify({
        # CNN result
        "cnn_class":      cnn_class,
        "class_label":    cfg["label"],
        "confidence":     round(cnn_conf * 100, 2),
        "recommendation": cfg["rec"],
        "all_scores": {
            CLASS_CFG[idx_to_name[i]]["label"]: round(float(p) * 100, 2)
            for i, p in enumerate(cnn_probs)
        },
        # Decision model
        "retake":         bool(retake_pred),
        "retake_prob":    round(retake_prob * 100, 2),
        # Regression
        "correction_pct": corr,
        "correction_dir": direction,
        # Physics
        "physics":        feat_dict,
        # Images
        "image_b64":      orig_b64,
        "gradcam_b64":    gradcam_b64,   # None if Grad-CAM failed
        "gradcam_error":  gradcam_error, # message shown in UI if failed
    })


# ENTRY POINT
if __name__ == "__main__":
    print(f"  Project root : {BASE}")
    print(f"  Models dir   : {MODELS_DIR}")
    print(f"  Open browser : http://localhost:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")