"""
VeriSync — Production Audio Expert Training
════════════════════════════════════════════
The audio task is ~balanced (1:1.1 ratio), so we use standard
Binary Cross-Entropy with label smoothing + cosine annealing.

Usage:
    python train_audio.py --prepare-splits --epochs 25 --batch-size 32
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_loader_audio import create_audio_dataset
from model_audio import create_audio_expert
from prepare_splits import create_splits


# ══════════════════════════════════════════════════════════════════
#  Cosine Annealing with Warmup (shared with visual trainer)
# ══════════════════════════════════════════════════════════════════

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
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
    parser = argparse.ArgumentParser(description="Train audio deepfake detector.")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--train-manifest", default="train_audio_manifest.csv")
    parser.add_argument("--val-manifest", default="val_audio_manifest.csv")
    parser.add_argument("--prepare-splits", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-model", default="audio_expert_best.keras")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _ensure_audio_labels(manifest_path):
    df = pd.read_csv(manifest_path)
    if "audio_label" in df.columns:
        df["label"] = df["audio_label"].astype(int)
    else:
        df["label"] = df["category"].apply(lambda x: 1 if "RealAudio" in str(x) else 0).astype(int)
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
            visual_fake_to_real_ratio=3.0,
            visual_train_out="train_manifest.csv",
            visual_val_out="val_manifest.csv",
            audio_train_out=args.train_manifest,
            audio_val_out=args.val_manifest,
        )

    train_df = _ensure_audio_labels(args.train_manifest)
    val_df = _ensure_audio_labels(args.val_manifest)

    print("\n" + "═" * 60)
    print("🎵 AUDIO EXPERT — TRAINING CONFIGURATION")
    print("═" * 60)
    print(f"  Architecture : EfficientNetB0 + Dense head")
    print(f"  Loss         : Binary Cross-Entropy (smoothing=0.05)")
    print(f"  LR Schedule  : Cosine Annealing with Warmup")
    print(f"  Train audios : {len(train_df):,}")
    print(f"  Val audios   : {len(val_df):,}")
    print(f"  Train labels : {train_df['label'].value_counts().to_dict()}")
    print(f"  Val labels   : {val_df['label'].value_counts().to_dict()}")

    class_weight = _compute_class_weights(train_df["label"].values)
    print(f"  Class weights: {class_weight}")
    print("═" * 60)

    # ── Datasets ──────────────────────────────────────────
    train_ds = create_audio_dataset(
        args.train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
    )
    val_ds = create_audio_dataset(
        args.val_manifest,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    # ── Model ─────────────────────────────────────────────
    model = create_audio_expert(input_shape=(128, 128, 1))

    # ── LR Schedule ───────────────────────────────────────
    steps_per_epoch = max(1, len(train_df) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * 2

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
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
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
        tf.keras.callbacks.CSVLogger("audio_training_log.csv"),
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
