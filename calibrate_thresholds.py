import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from data_loader import create_visual_dataset
from data_loader_audio import create_audio_dataset


def _prepare_eval_manifest(manifest_path, label_col):
    df = pd.read_csv(manifest_path).copy()
    if label_col in df.columns:
        df["label"] = df[label_col].astype(int)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name, df["label"].values.astype(int)


def _find_best_threshold(y_true, y_score):
    candidates = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, best_f1


def _compute_eer(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    if np.all(np.isnan(fpr)) or np.all(np.isnan(fnr)):
        return float("nan"), 0.5
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def _evaluate(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        best_t, best_f1 = _find_best_threshold(y_true, y_score)
        y_pred = (y_score >= best_t).astype(int)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "best_f1": float(best_f1),
            "best_threshold": float(best_t),
            "balanced_acc_at_best_f1": bal_acc,
            "eer": float("nan"),
            "eer_threshold": 0.5,
        }

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    best_t, best_f1 = _find_best_threshold(y_true, y_score)
    y_pred = (y_score >= best_t).astype(int)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    eer, eer_t = _compute_eer(y_true, y_score)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": float(best_f1),
        "best_threshold": float(best_t),
        "balanced_acc_at_best_f1": bal_acc,
        "eer": eer,
        "eer_threshold": eer_t,
    }


def _load_model(path_options):
    for path in path_options:
        try:
            return tf.keras.models.load_model(path), path
        except Exception:
            continue
    raise FileNotFoundError(f"Could not load model from options: {path_options}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate thresholds on validation sets.")
    parser.add_argument("--visual-model", default="visual_expert_best.keras")
    parser.add_argument("--audio-model", default="audio_expert_best.keras")
    parser.add_argument("--visual-manifest", default="val_manifest.csv")
    parser.add_argument("--audio-manifest", default="val_audio_manifest.csv")
    parser.add_argument("--visual-batch-size", type=int, default=8)
    parser.add_argument("--audio-batch-size", type=int, default=32)
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()

    visual_model, visual_path = _load_model([args.visual_model, "visual_expert_best.h5"])
    audio_model, audio_path = _load_model([args.audio_model, "audio_expert_best.h5"])

    visual_manifest, yv = _prepare_eval_manifest(args.visual_manifest, "video_label")
    audio_manifest, ya = _prepare_eval_manifest(args.audio_manifest, "audio_label")

    visual_ds = create_visual_dataset(
        visual_manifest,
        batch_size=args.visual_batch_size,
        shuffle=False,
        augment=False,
    )
    audio_ds = create_audio_dataset(
        audio_manifest,
        batch_size=args.audio_batch_size,
        shuffle=False,
        augment=False,
    )

    pv = visual_model.predict(visual_ds, verbose=1).reshape(-1)
    pa = audio_model.predict(audio_ds, verbose=1).reshape(-1)

    visual_metrics = _evaluate(yv, pv)
    audio_metrics = _evaluate(ya, pa)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visual_model_path": visual_path,
        "audio_model_path": audio_path,
        "visual_real_threshold": visual_metrics["best_threshold"],
        "audio_real_threshold": audio_metrics["best_threshold"],
        "fusion_real_threshold": 0.5,
        "visual_metrics": visual_metrics,
        "audio_metrics": audio_metrics,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Calibration complete.")
    print(f"Saved calibration to: {args.output}")
    print("Visual metrics:", visual_metrics)
    print("Audio metrics:", audio_metrics)

    Path(visual_manifest).unlink(missing_ok=True)
    Path(audio_manifest).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
