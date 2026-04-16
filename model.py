"""
VeriSync — Visual Expert Model
══════════════════════════════
Simple and proven architecture: EfficientNetB0 → LSTM → Dense head
Input: (T, 224, 224, 3) face crop sequence
Output: P(real) ∈ [0, 1]
"""

import ssl
import tensorflow as tf
from tensorflow.keras import Model, layers

ssl._create_default_https_context = ssl._create_unverified_context


def create_visual_expert(
    input_shape=(10, 224, 224, 3),
    lstm_units=128,
    train_backbone=True,
):
    """
    EfficientNetB0 + LSTM deepfake detector.
    - EfficientNetB0 extracts per-frame spatial features
    - LSTM captures temporal inconsistencies across frames
    - Dense head outputs P(real)
    """
    sequence_len, image_size = input_shape[0], input_shape[1]
    inputs = layers.Input(shape=input_shape, name="video_input")

    # ── Backbone: EfficientNetB0 (pretrained) ─────────────
    try:
        base_cnn = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )
    except Exception as exc:
        print(f"Warning: imagenet weights failed ({exc}), using random init.")
        base_cnn = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )

    if train_backbone:
        # Fine-tune only the last ~30 layers, freeze early features
        for layer in base_cnn.layers[:-30]:
            layer.trainable = False
    else:
        base_cnn.trainable = False

    # ── Per-frame feature extraction ──────────────────────
    x = layers.TimeDistributed(base_cnn, name="frame_encoder")(inputs)
    x = layers.LayerNormalization(name="frame_ln")(x)

    # ── Temporal modeling: LSTM ───────────────────────────
    x = layers.LSTM(lstm_units, return_sequences=False, dropout=0.3, name="temporal_lstm")(x)

    # ── Classification head ──────────────────────────────
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs, name="VisualExpert_EffNet_LSTM")


if __name__ == "__main__":
    model = create_visual_expert()
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")