"""
MODULE 5 — Grad-CAM Explainability
PURPOSE: Generate a heatmap showing WHICH regions of the X-ray most influenced the CNN's exposure classification.
"""

import numpy as np
import cv2
import tensorflow as tf


def _find_last_conv_layer(model):
    """
    Automatically find the last Conv2D layer in the model.
    For MobileNetV2, this is typically 'Conv_1' or similar.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # For MobileNetV2 wrapped as a sub-model
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    return None


def generate_gradcam(model, img_array, class_index,
                     layer_name=None, alpha=0.4):
    """
    Generate a Grad-CAM heatmap and overlay it on the input image.

    Parameters:
      model       : trained Keras model (full classification model)
      img_array   : numpy array shape (1, 224, 224, 3), preprocessed
                    (MobileNetV2 preprocess_input applied)
      class_index : integer index of the predicted class
      layer_name  : name of the conv layer to use (auto-detected if None)
      alpha       : blending factor for overlay (0=no heatmap, 1=full)

    Returns:
      overlay_bgr : (224, 224, 3) uint8 BGR image with heatmap overlay
      heatmap     : (224, 224) float32 raw heatmap (0-1)
    """

    # ── Find target layer ─────────────────────────────────────
    if layer_name is None:
        # For MobileNetV2, use the last conv layer in the base
        layer_name = "Conv_1"   # MobileNetV2 last conv block output
        # Verify it exists
        layer_names = [l.name for l in model.layers]
        if layer_name not in layer_names:
            # Try to find it inside the MobileNetV2 sub-model
            for l in model.layers:
                if hasattr(l, "layers"):
                    inner = [sl.name for sl in l.layers if "Conv_1" in sl.name]
                    if inner:
                        layer_name = inner[-1]
                        break

    # ── Build gradient model ──────────────────────────────────
    # This model outputs both the feature maps of the target layer
    # AND the final class probabilities.
    try:
        # Try direct layer access first
        target_layer = model.get_layer(layer_name)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
    except ValueError:
        # MobileNetV2 is a sub-model — navigate inside it
        base_model   = None
        for l in model.layers:
            if "mobilenet" in l.name.lower():
                base_model = l
                break
        if base_model is None:
            raise ValueError("Cannot find MobileNetV2 layer in model.")

        target_layer = base_model.get_layer(layer_name)
        # Build intermediate model through the base
        inner_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[target_layer.output, base_model.output]
        )
        # Wrap: full model → (conv_output, final_output)
        # We use GradientTape on the full model instead
        grad_model = None   # use tape path below

    # ── Compute gradients ─────────────────────────────────────
    with tf.GradientTape() as tape:
        if grad_model is not None:
            conv_outputs, predictions = grad_model(img_array)
        else:
            # Tape path for nested models
            tape.watch(img_array)
            conv_outputs, _ = inner_model(img_array)
            predictions = model(img_array)
        tape.watch(conv_outputs)

        # Score for the predicted class
        class_score = predictions[:, class_index]

    # Gradient of class score w.r.t. conv feature maps
    grads = tape.gradient(class_score, conv_outputs)

    # ── Pool gradients over spatial dimensions ────────────────
    # Shape: (num_filters,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each channel of the conv output by its importance
    conv_out = conv_outputs[0]                          # (H, W, C)
    heatmap  = conv_out @ pooled_grads[..., tf.newaxis] # (H, W, 1)
    heatmap  = tf.squeeze(heatmap)                      # (H, W)

    # ── Normalise and apply ReLU ──────────────────────────────
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # ── Resize to original image size ─────────────────────────
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # ── Colormap: jet (blue=low, red=high influence) ──────────
    heatmap_uint8  = np.uint8(255 * heatmap_resized)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # ── Convert original image to display range ───────────────
    # img_array is in MobileNetV2 range [-1, 1] → bring back to [0,255]
    orig = img_array[0].numpy() if hasattr(img_array[0], "numpy") \
           else img_array[0]
    orig_display = ((orig + 1.0) / 2.0 * 255).astype(np.uint8)
    orig_bgr     = cv2.cvtColor(orig_display, cv2.COLOR_RGB2BGR)

    # ── Blend heatmap over original X-ray ─────────────────────
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha,
                              heatmap_color, alpha, 0)

    return overlay, heatmap_resized


def save_gradcam(overlay_bgr, save_path):
    """Save the Grad-CAM overlay to a file."""
    cv2.imwrite(save_path, overlay_bgr)


def gradcam_to_base64(overlay_bgr):
    """Convert Grad-CAM overlay to base64 PNG string for Flask."""
    import base64
    _, buf = cv2.imencode(".png", overlay_bgr)
    return base64.b64encode(buf).decode("utf-8")
