"""
Grad-CAM heatmap generation using the trained DR classifier.
Replaces the old U-Net segmentation module.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D


# ── Compat fix for Keras 3 deserialization ────────────────────────────────────
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)


# ── Module-level model cache ─────────────────────────────────────────────────
_model = None


def _load_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        print("Loading trained DR model for Grad-CAM...")
        _model = load_model(
            "models/dr_model_trained.h5",
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
        )
    return _model


def _find_last_conv_layer(model):
    """Return the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(image_path):
    """
    Generate a Grad-CAM heatmap overlay for the given retinal image.

    Returns
    -------
    gradcam_path : str
        Relative path to the saved heatmap image (static/masks/gradcam_<stem>.jpg)
    """
    model = _load_model()
    os.makedirs("static/masks", exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = preprocess_input(img_resized.astype(np.float32))
    img_tensor = tf.cast(img_array[np.newaxis, ...], tf.float32)  # (1,224,224,3)

    # ── 2. Identify the last conv layer ───────────────────────────────────────
    last_conv_name = _find_last_conv_layer(model)
    last_conv_layer = model.get_layer(last_conv_name)

    # ── 3. Quick forward pass to find predicted class index ───────────────────
    preds_np = model.predict(img_tensor, verbose=0)  # always returns np array
    pred_index = int(np.argmax(preds_np[0]))

    # ── 4. Build a gradient model and compute Grad-CAM ────────────────────────
    #    We create a tf.function to avoid issues with eager vs graph mode.
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        # Forward pass inside tape
        inputs = tf.cast(img_tensor, tf.float32)
        outputs = grad_model(inputs, training=False)
        # outputs may be a list; index into it safely
        conv_outputs = outputs[0]
        predictions = outputs[1]
        tape.watch(conv_outputs)
        # Recompute class_score from predictions that tape tracks
        class_score = predictions[:, pred_index]

    # Compute gradients of the predicted class w.r.t. conv layer output
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        # Fallback: if gradients can't be computed, produce a neutral heatmap
        print("[Grad-CAM] WARNING: gradients are None, producing neutral heatmap")
        h_orig, w_orig = img_bgr.shape[:2]
        heatmap_color = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        overlay = img_bgr.copy()
    else:
        # ── 5. Pool gradients → weighted feature map ──────────────────────────
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)
        conv_out = conv_outputs[0]                                # (H, W, C)
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]       # (H, W, 1)
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalize to [0, 1]
        heatmap = tf.maximum(heatmap, 0.0)
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max

        heatmap_np = heatmap.numpy()

        # ── 6. Resize heatmap to original image size ──────────────────────────
        h_orig, w_orig = img_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap_np, (w_orig, h_orig))

        # Apply JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # ── 7. Blend heatmap with original image ─────────────────────────────
        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap_color, 0.45, 0)

    # ── 8. Save output ────────────────────────────────────────────────────────
    base_name = os.path.basename(image_path)
    stem, _ = os.path.splitext(base_name)
    out_filename = f"gradcam_{stem}.jpg"
    gradcam_path = os.path.join("static", "masks", out_filename)

    cv2.imwrite(gradcam_path, overlay)
    print(f"[Grad-CAM] Heatmap saved -> {gradcam_path}")

    return gradcam_path


# ── Backward-compat shim ─────────────────────────────────────────────────────
def predict_dr(image_path):
    gradcam_path = generate_gradcam(image_path)
    return "GradCAM", gradcam_path