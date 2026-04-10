"""
VeriSync — Production Visual Expert Training
═════════════════════════════════════════════
Key techniques for the 1:13 class imbalance:
  • Focal Loss (γ=2.0) — focuses on hard examples
  • Label Smoothing (0.05) — prevents overconfidence
  • Cosine Annealing with Warmup — stable LR schedule
  • Balanced Class Weights — mathematically computed
  • Controlled fake:real ratio (3:1 cap)
  • Strong augmentation pipeline

Usage:
    python train_visual.py --prepare-splits --epochs 20 --batch-size 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_loader import create_visual_dataset
from model import create_visual_expert
from prepare_splits import create_splits


# ══════════════════════════════════════════════════════════════════
#  Focal Loss — critical for 1:13 imbalanced data
# ══════════════════════════════════════════════════════════════════

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for binary classification with class imbalance.

    FL(p) = -α * (1-p)^γ * log(p)    for y=1
    FL(p) = -(1-α) * p^γ * log(1-p)  for y=0

    γ (gamma) = 2.0: Focus more on hard-to-classify examples
    α (alpha) = 0.75: Give more weight to the minority class (real videos)
    """
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Clip for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Focal loss computation
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        return tf.reduce_mean(alpha_t * focal_weight * bce)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "label_smoothing": self.label_smoothing})
        return config


# ══════════════════════════════════════════════════════════════════
#  Cosine Annealing with Warmup
# ══════════════════════════════════════════════════════════════════

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup for first N steps, then cosine decay to min_lr.
    Helps with stable training on pretrained backbones.
    """
    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))

        decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(
            np.pi * (step - self.warmup_steps) / decay_steps
        ))
        decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": int(self.warmup_steps),
            "total_steps": int(self.total_steps),
            "min_lr": self.min_lr,
        }


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(description="Train visual deepfake detector.")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--train-manifest", default="train_manifest.csv")
    parser.add_argument("--val-manifest", default="val_manifest.csv")
    parser.add_argument("--prepare-splits", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fake-to-real-ratio", type=float, default=3.0,
                        help="Cap fake samples to N × real samples to reduce imbalance")
    parser.add_argument("--output-model", default="visual_expert_best.keras")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _ensure_visual_labels(manifest_path):
    df = pd.read_csv(manifest_path)
    if "video_label" in df.columns:
        df["label"] = df["video_label"].astype(int)
        df.to_csv(manifest_path, index=False)
    return df


def _compute_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def main():
    args = _parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_manifest_path = Path(args.train_manifest)
    val_manifest_path = Path(args.val_manifest)

    if args.prepare_splits or not train_manifest_path.exists() or not val_manifest_path.exists():
        print("Creating leakage-safe splits...")
        create_splits(
            manifest_path=args.manifest,
            seed=args.seed,
            visual_val_ratio=0.2,
            audio_val_ratio=0.15,
            visual_fake_to_real_ratio=args.fake_to_real_ratio,
            visual_train_out=args.train_manifest,
            visual_val_out=args.val_manifest,
            audio_train_out="train_audio_manifest.csv",
            audio_val_out="val_audio_manifest.csv",
        )

    train_df = _ensure_visual_labels(args.train_manifest)
    val_df = _ensure_visual_labels(args.val_manifest)

    print("\n" + "═" * 60)
    print("🧠 VISUAL EXPERT — TRAINING CONFIGURATION")
    print("═" * 60)
    print(f"  Architecture : EfficientNetB0 + LSTM")
    print(f"  Loss         : Focal Loss (γ=2.0, α=0.75, smoothing=0.05)")
    print(f"  LR Schedule  : Cosine Annealing with Warmup")
    print(f"  Train videos : {len(train_df):,}")
    print(f"  Val videos   : {len(val_df):,}")
    print(f"  Train labels : {train_df['label'].value_counts().to_dict()}")
    print(f"  Val labels   : {val_df['label'].value_counts().to_dict()}")

    class_weight = _compute_class_weights(train_df["label"].values)
    print(f"  Class weights: {class_weight}")
    print("═" * 60)

    # ── Datasets ──────────────────────────────────────────
    train_ds = create_visual_dataset(
        args.train_manifest,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        shuffle=True,
        augment=True,
    )
    val_ds = create_visual_dataset(
        args.val_manifest,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        shuffle=False,
        augment=False,
    )

    # ── Model ─────────────────────────────────────────────
    model = create_visual_expert(input_shape=(args.num_frames, 224, 224, 3))

    # ── LR Schedule ───────────────────────────────────────
    steps_per_epoch = max(1, len(train_df) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * 2  # 2-epoch warmup

    lr_schedule = WarmupCosineDecay(
        initial_lr=args.learning_rate,
        warmup_steps=float(warmup_steps),
        total_steps=float(total_steps),
        min_lr=1e-7,
    )

    # ── Compile ───────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-5,
        ),
        loss=FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="auprc", curve="PR"),
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
        ],
    )

    # ── Callbacks ─────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output_model,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger("visual_training_log.csv"),
    ]

    # ── Train ─────────────────────────────────────────────
    print("\n🚀 Starting training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    best_auc = max(history.history.get("val_auc", [0.0]))
    best_acc = max(history.history.get("val_acc", [0.0]))
    print(f"\n{'═' * 60}")
    print(f"✅ Training complete!")
    print(f"   Best val_auc : {best_auc:.4f}")
    print(f"   Best val_acc : {best_acc:.4f}")
    print(f"   Saved to     : {args.output_model}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
