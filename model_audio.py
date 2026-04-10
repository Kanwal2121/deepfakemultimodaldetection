"""
VeriSync — Audio Expert Model
═════════════════════════════
EfficientNetB0 on mel-spectrograms (converted to 3-channel pseudo-RGB).
Input: (128, 128, 1) mel-spectrogram
Output: P(real audio) ∈ [0, 1]
"""

import ssl
import tensorflow as tf
from tensorflow.keras import Model, layers

ssl._create_default_https_context = ssl._create_unverified_context


def create_audio_expert(
    input_shape=(128, 128, 1),
    train_backbone=True,
):
    """
    EfficientNetB0-based audio deepfake detector.
    Converts single-channel spectrograms to pseudo-RGB for transfer learning.
    """
    inputs = layers.Input(shape=input_shape, name="audio_input")

    # Convert 1-channel spectrogram → 3-channel for pretrained backbone
    x = layers.Concatenate(name="to_rgb")([inputs, inputs, inputs])

    # ── Backbone: EfficientNetB0 ──────────────────────────
    try:
        base_cnn = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling="avg",
        )
    except Exception as exc:
        print(f"Warning: imagenet weights failed ({exc}), using random init.")
        base_cnn = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling="avg",
        )

    if train_backbone:
        for layer in base_cnn.layers[:-30]:
            layer.trainable = False
    else:
        base_cnn.trainable = False

    x = base_cnn(x)

    # ── Classification head ──────────────────────────────
    x = layers.LayerNormalization(name="audio_ln")(x)
    x = layers.Dense(128, activation="relu", name="audio_dense")(x)
    x = layers.Dropout(0.4, name="audio_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs, name="AudioExpert_EffNet")


if __name__ == "__main__":
    model = create_audio_expert()
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")
