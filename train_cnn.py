"""
MODULE 2 — CNN Model (MobileNetV2 Fine-tuning)
PURPOSE: Train a MobileNetV2-based CNN to classify X-ray images into underexposed / proper / overexposed.
WHY MobileNetV2:
  - Pretrained on ImageNet: already knows edges, textures, shapes
  - Depthwise separable convolutions: fast inference
  - Transfer learning: needs fewer X-ray images to converge (1000 original in our case)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

BASE      = r"D:\AI_Final_Project"
DATASET_DIR  = os.path.join(BASE, "data", "processed")
MODELS_DIR   = os.path.join(BASE, "models")
MODEL_SAVE   = os.path.join(MODELS_DIR, "cnn_model.keras")
INDICES_SAVE = os.path.join(MODELS_DIR, "class_indices.json")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 3

os.makedirs(MODELS_DIR, exist_ok=True)


# DATA GENERATORS
def build_generators():
    """
    MobileNetV2 expects inputs in [-1, 1] via preprocess_input.
    We use ImageDataGenerator with the MobileNetV2 preprocessor.
    Augmentation on train only — test data must be unmodified.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=8,
        zoom_range=0.08,
        width_shift_range=0.05,
        height_shift_range=0.05,
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"  Class indices: {train_gen.class_indices}")
    with open(INDICES_SAVE, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"  Saved → {INDICES_SAVE}")

    return train_gen, test_gen

# MODEL ARCHITECTURE
def build_model():
    """
    Architecture:
      MobileNetV2 base (frozen in Phase 1)
        → GlobalAveragePooling2D        (spatial → vector)
        → Dense(256, relu)              (high-level features)
        → BatchNormalization            (training stability)
        → Dropout(0.4)                  (prevent overfitting)
        → Dense(128, relu)  ← FEATURE EXTRACTION LAYER
        → BatchNormalization
        → Dropout(0.3)
        → Dense(3, softmax)             (class probabilities)

    The Dense(128) layer output is the feature vector
    used by the decision and regression models.
    """
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Frozen in Phase 1

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    features = layers.Dense(128, activation="relu", name="feature_vector")(x)
    x = layers.BatchNormalization()(features)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inputs, output, name="XrayMobileNetV2")
    return model, base


# TWO-PHASE TRAINING

def phase1_train(model, train_gen, test_gen):
    """Phase 1: Train only the classification head. LR=1e-3."""
    print("\n  ── Phase 1: Training head (base frozen) ──")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4,
                          restore_best_weights=True),
        ],
        verbose=1,
    )
    return history


def phase2_train(model, base, train_gen, test_gen):
    """
    Phase 2: Unfreeze top 30 layers of MobileNetV2.
    Use very low LR (1e-5) to fine-tune without destroying
    the pretrained ImageNet features.
    """
    print("\n  ── Phase 2: Fine-tuning top 30 base layers ──")

    # Unfreeze top 30 layers only
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    # Must recompile after changing trainability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=25,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-7, verbose=1),
        ],
        verbose=1,
    )
    return history

# EVALUATION

def evaluate(model, test_gen):
    with open(INDICES_SAVE) as f:
        name_to_idx = json.load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    class_names = [idx_to_name[i] for i in range(NUM_CLASSES)]

    # Collect predictions
    all_true, all_pred = [], []
    for i in range(len(test_gen)):
        imgs, labels = test_gen[i]
        preds = model.predict(imgs, verbose=0)
        all_true.extend(np.argmax(labels, axis=1))
        all_pred.extend(np.argmax(preds,  axis=1))

    print("\n  Classification Report:")
    print(classification_report(all_true, all_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("CNN Confusion Matrix")
    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Confusion matrix → {out}")


def plot_histories(h1, h2):
    acc  = h1.history["accuracy"]      + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"]  + h2.history["val_accuracy"]
    loss = h1.history["loss"]          + h2.history["loss"]
    vloss= h1.history["val_loss"]      + h2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(acc,  label="Train", color="#2196F3")
    ax1.plot(vacc, label="Val",   color="#4CAF50")
    ax1.axvline(len(h1.history["accuracy"])-1, color="gray",
                linestyle="--", alpha=0.6, label="Phase 2 start")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(loss,  label="Train", color="#F44336")
    ax2.plot(vloss, label="Val",   color="#FF9800")
    ax2.axvline(len(h1.history["loss"])-1, color="gray",
                linestyle="--", alpha=0.6)
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "training_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Training curves → {out}")

# MAIN
if __name__ == "__main__":
    print("=" * 52)
    print("  MODULE 2 — MobileNetV2 CNN Training")
    print("=" * 52)

    if not os.path.exists(os.path.join(DATASET_DIR, "train")):
        print("  Run prepare_data.py first!")
        exit(1)

    train_gen, test_gen = build_generators()
    model, base = build_model()
    model.summary()

    h1 = phase1_train(model, train_gen, test_gen)
    h2 = phase2_train(model, base, train_gen, test_gen)

    plot_histories(h1, h2)
    evaluate(model, test_gen)

    print(f"\n  Model saved → {MODEL_SAVE}")
    print("  Next: python scripts/train_decision_model.py")
