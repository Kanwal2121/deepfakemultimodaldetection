"""
Grad-CAM Explainability Module for VeriSync Visual Expert.

Generates heatmaps highlighting the facial regions the model considers
most indicative of manipulation.  Works with the TimeDistributed
EfficientNet architecture used in model.py.
"""

import cv2
import numpy as np
import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────
#  Core Grad-CAM
# ──────────────────────────────────────────────────────────────────────

def _find_last_conv_layer(model):
    """Walk *all* nested layers (including inside TimeDistributed) to
    locate the last Conv2D layer in the backbone."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TimeDistributed):
            inner = layer.layer
            if isinstance(inner, tf.keras.Model):
                for sub in inner.layers:
                    if isinstance(sub, (tf.keras.layers.Conv2D,)):
                        last_conv = (layer.name, sub.name)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = (layer.name, None)
    return last_conv


def _build_gradcam_model(model):
    """Build a sub-model that outputs (conv_output, prediction) so we
    can compute gradients of the prediction w.r.t. the conv feature map."""
    # Strategy: extract the frame_encoder (TimeDistributed CNN) output
    # and the final prediction in one forward pass.
    frame_encoder = None
    for layer in model.layers:
        if layer.name == "frame_encoder":
            frame_encoder = layer
            break

    if frame_encoder is None:
        raise ValueError("Could not find 'frame_encoder' layer.")

    # The inner CNN model inside TimeDistributed
    cnn = frame_encoder.layer
    last_conv_layer = None
    for layer in cnn.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
    if last_conv_layer is None:
        raise ValueError("No Conv2D found inside the CNN backbone.")

    return cnn, last_conv_layer.name


def compute_gradcam_heatmaps(model, frames_batch, class_idx=0):
    """
    Compute Grad-CAM heatmaps for each frame in a video sequence.

    Parameters
    ----------
    model : tf.keras.Model
        The full VisualExpertV2 model.
    frames_batch : np.ndarray
        Shape (1, T, H, W, 3) — a single video's frames.
    class_idx : int
        Class index (0 = output neuron for P(real)).

    Returns
    -------
    heatmaps : list[np.ndarray]
        List of T heatmaps, each of shape (H, W) in range [0, 1].
    """
    cnn, last_conv_name = _build_gradcam_model(model)

    # Build a mini-model that returns last-conv activations + final pool
    conv_output_model = tf.keras.Model(
        cnn.input,
        outputs=[cnn.get_layer(last_conv_name).output, cnn.output],
    )

    frames = frames_batch[0]  # (T, H, W, 3)
    T = frames.shape[0]
    heatmaps = []

    for t in range(T):
        single_frame = tf.cast(frames[t:t+1], tf.float32)  # (1, H, W, 3)
        processed = single_frame
        
        with tf.GradientTape() as tape:
            tape.watch(processed)
            conv_out, pool_out = conv_output_model(processed, training=False)
            tape.watch(conv_out)
            # Use the pooled output as a proxy for the prediction contribution
            # of this frame (the full temporal model isn't differentiable
            # per-frame easily, so we approximate).
            target = tf.reduce_mean(pool_out)

        grads = tape.gradient(target, conv_out)  # (1, h, w, C)
        if grads is None:
            heatmaps.append(np.zeros((frames.shape[1], frames.shape[2]), dtype=np.float32))
            continue

        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # (1, 1, 1, C)
        cam = tf.reduce_sum(weights * conv_out, axis=-1)[0]  # (h, w)
        cam = tf.nn.relu(cam).numpy()

        # Normalize
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Resize to original frame dimensions
        cam_resized = cv2.resize(cam, (frames.shape[2], frames.shape[1]))
        heatmaps.append(cam_resized)

    return heatmaps


# ──────────────────────────────────────────────────────────────────────
#  Visualization helpers
# ──────────────────────────────────────────────────────────────────────

def overlay_heatmap(frame, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Blend a Grad-CAM heatmap onto a frame image.

    Parameters
    ----------
    frame : np.ndarray
        (H, W, 3) in [0, 255] uint8 or float32.
    heatmap : np.ndarray
        (H, W) in [0, 1] float.
    alpha : float
        Opacity of the heatmap overlay.

    Returns
    -------
    blended : np.ndarray
        (H, W, 3) uint8 image with heatmap overlay.
    """
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
    return blended


def generate_gradcam_gallery(model, frames_batch):
    """
    Returns a list of (original_frame, heatmap_overlay) tuples for display.
    """
    heatmaps = compute_gradcam_heatmaps(model, frames_batch)
    frames = frames_batch[0]  # (T, H, W, 3)
    gallery = []
    for t, hm in enumerate(heatmaps):
        frame = frames[t]
        overlay = overlay_heatmap(frame, hm, alpha=0.45)
        gallery.append((frame, overlay, hm))
    return gallery


# ──────────────────────────────────────────────────────────────────────
#  CLI test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from preprocess_utils import extract_and_crop_faces

    model_path = "visual_expert_best.keras"
    if not os.path.exists(model_path):
        model_path = "visual_expert_besttt.h5"
    if not os.path.exists(model_path):
        model_path = "visual_expert_best.h5"

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Use a sample video from the manifest
    import pandas as pd
    df = pd.read_csv("data_manifest.csv")
    sample_path = df.iloc[0]["video_path"]
    print(f"Processing: {sample_path}")

    faces = extract_and_crop_faces(sample_path, num_frames=10)
    faces_batch = np.expand_dims(faces, axis=0)

    gallery = generate_gradcam_gallery(model, faces_batch)
    print(f"Generated {len(gallery)} Grad-CAM overlays.")

    # Save a sample
    for i, (orig, overlay, hm) in enumerate(gallery[:3]):
        out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"gradcam_frame_{i}.png", out)
        print(f"Saved gradcam_frame_{i}.png")
