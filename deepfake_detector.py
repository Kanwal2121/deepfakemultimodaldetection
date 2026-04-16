"""
VeriSync — Multi-Modal Deepfake Detector (CLI)
═══════════════════════════════════════════════
Enhanced with:
  • Frequency domain analysis (4th modality)
  • Confidence-weighted adaptive fusion
  • Richer forensic report

Usage:
    python deepfake_detector.py <video_path> [--json-out report.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from frequency_analyzer import analyze_video_frequency
from lip_sync_analyzer import LipSyncAnalyzer
from preprocess_utils import extract_and_crop_faces, extract_mel_spectrogram


# ══════════════════════════════════════════════════════════════════
#  Custom Keras Objects (Required for loading pre-trained models)
# ══════════════════════════════════════════════════════════════════

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        return tf.reduce_mean(alpha_t * focal_weight * bce)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha, "label_smoothing": self.label_smoothing})
        return config


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr=1e-7, **kwargs):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * (step - self.warmup_steps) / decay_steps))
        decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {"initial_lr": self.initial_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps, "min_lr": self.min_lr}


DEFAULT_CALIBRATION = {
    "visual_real_threshold": 0.5,
    "audio_real_threshold": 0.5,
    "fusion_real_threshold": 0.5,
    "lip_sync_real_threshold": 0.61,
}


def _load_model(path_options):
    for path in path_options:
        if not os.path.exists(path):
            continue
        try:
            custom_objects = {"FocalLoss": FocalLoss, "WarmupCosineDecay": WarmupCosineDecay}
            return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False), path
        except Exception:
            continue

    raise FileNotFoundError(f"Unable to load model from: {path_options}")


def _load_calibration(path="calibration.json"):
    if not os.path.exists(path):
        return DEFAULT_CALIBRATION.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = DEFAULT_CALIBRATION.copy()
        cfg.update(raw)
        return cfg
    except Exception:
        return DEFAULT_CALIBRATION.copy()


def _confidence(p):
    """Entropy-based confidence: 1.0 = maximally confident, 0.0 = uncertain."""
    eps = 1e-7
    p_c = np.clip(p, eps, 1 - eps)
    entropy = -(p_c * np.log2(p_c) + (1 - p_c) * np.log2(1 - p_c))
    return float(1.0 - entropy)


print("Loading forensic experts...")
try:
    visual_model, visual_model_path = _load_model([
        "visual_expert_best.keras", "visual_expert_best.h5", "visual_expert_besttt.h5",
    ])
    audio_model, audio_model_path = _load_model([
        "audio_expert_best.keras", "audio_expert_best.h5",
    ])
    sync_analyzer = LipSyncAnalyzer()
    calibration = _load_calibration("calibration.json")
except Exception as e:
    print(f"Error loading experts: {e}")
    sys.exit(1)


def detect_deepfake(video_path, json_out=None):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    print(f"Analyzing: {os.path.basename(video_path)}")

    # 1) Visual Expert
    faces = extract_and_crop_faces(video_path, num_frames=10)
    faces_batch = np.expand_dims(faces, axis=0)
    vis_real_prob = float(visual_model.predict(faces_batch, verbose=0)[0][0])
    vis_thr = float(calibration.get("visual_real_threshold", 0.5))
    vis_verdict = "REAL" if vis_real_prob >= vis_thr else "FAKE"
    vis_conf = _confidence(vis_real_prob)

    # 2) Audio Expert
    spec = extract_mel_spectrogram(video_path)
    is_silent = bool(np.all(spec == 0.5))
    aud_real_prob = 0.5
    aud_verdict = "N/A"
    aud_conf = 0.0
    aud_thr = float(calibration.get("audio_real_threshold", 0.5))
    if not is_silent:
        spec_batch = np.expand_dims(spec, axis=0)
        aud_real_prob = float(audio_model.predict(spec_batch, verbose=0)[0][0])
        aud_verdict = "REAL" if aud_real_prob >= aud_thr else "FAKE"
        aud_conf = _confidence(aud_real_prob)

    # 3) Lip-Sync Expert
    sync_score = 0.5
    sync_verdict = "N/A"
    sync_thr = float(calibration.get("lip_sync_real_threshold", 0.61))
    if not is_silent:
        sync_estimate = sync_analyzer.compute_sync_score(video_path)
        if sync_estimate is not None:
            sync_score = float(sync_estimate)
            sync_verdict = "REAL" if sync_score >= sync_thr else "FAKE"

    # 4) Frequency Domain Expert (NEW)
    freq_summary, freq_per_frame = analyze_video_frequency(faces)
    freq_anomaly = freq_summary.get("anomaly_score", 0.5)
    freq_verdict = "CLEAN" if freq_anomaly < 0.45 else "SUSPECT"

    # 5) Confidence-Weighted Adaptive Fusion (ENHANCED)
    if is_silent:
        # No audio signals available — rely on visual + frequency
        fused_real_prob = 0.75 * vis_real_prob + 0.25 * (1.0 - freq_anomaly)
    else:
        # All 4 experts contribute, weighted by their confidence
        # All 4 experts contribute, weighted by their confidence
        conf_s = _confidence(sync_score)
        conf_f = _confidence(1.0 - freq_anomaly)
        total_conf = vis_conf + aud_conf + conf_s + conf_f + 1e-8
        
        w_v = vis_conf / total_conf
        w_a = aud_conf / total_conf
        w_s = conf_s / total_conf
        w_f = conf_f / total_conf


        fused_real_prob = (
            w_v * vis_real_prob
            + w_a * aud_real_prob
            + w_s * sync_score
            + w_f * (1.0 - freq_anomaly)
        )

    fused_thr = float(calibration.get("fusion_real_threshold", 0.5))
    final_verdict = "REAL" if fused_real_prob >= fused_thr else "FAKE"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": video_path,
        "models": {
            "visual": visual_model_path,
            "audio": audio_model_path,
        },
        "thresholds": {
            "visual_real_threshold": vis_thr,
            "audio_real_threshold": aud_thr,
            "fusion_real_threshold": fused_thr,
            "lip_sync_real_threshold": sync_thr,
        },
        "scores": {
            "visual_real_prob": vis_real_prob,
            "visual_confidence": vis_conf,
            "audio_real_prob": aud_real_prob,
            "audio_confidence": aud_conf,
            "lip_sync_score": sync_score,
            "frequency_anomaly": freq_anomaly,
            "fused_real_prob": fused_real_prob,
        },
        "verdicts": {
            "visual": vis_verdict,
            "audio": aud_verdict,
            "lip_sync": sync_verdict,
            "frequency": freq_verdict,
            "final": final_verdict,
        },
        "frequency_features": {
            "high_freq_ratio": freq_summary.get("high_freq_ratio"),
            "spectral_flatness": freq_summary.get("spectral_flatness"),
            "dct_kurtosis": freq_summary.get("dct_kurtosis"),
            "rolloff_slope": freq_summary.get("rolloff_slope"),
        },
        "flags": {
            "is_silent_audio": is_silent,
        },
    }

    # ── Console Report ────────────────────────────────────
    print("\n" + "═" * 62)
    print("  🔬 VERISYNC MULTI-MODAL FORENSIC REPORT")
    print("═" * 62)
    print(f"  VISUAL    : {vis_verdict:>7} │ P(real)={vis_real_prob:.3f} │ conf={vis_conf:.2f} │ thr={vis_thr:.3f}")
    if not is_silent:
        print(f"  AUDIO     : {aud_verdict:>7} │ P(real)={aud_real_prob:.3f} │ conf={aud_conf:.2f} │ thr={aud_thr:.3f}")
        print(f"  LIP-SYNC  : {sync_verdict:>7} │ score ={sync_score:.3f} │            │ thr={sync_thr:.3f}")
    else:
        print("  AUDIO     :     N/A │ silent audio detected")
        print("  LIP-SYNC  :     N/A │ skipped (silent audio)")
    print(f"  FREQUENCY : {freq_verdict:>7} │ anomaly={freq_anomaly:.3f}")
    print("─" * 62)
    emoji = "✅" if final_verdict == "REAL" else "🚨"
    print(f"  {emoji} FINAL    : {final_verdict:>7} │ P(real)={fused_real_prob:.3f} │ thr={fused_thr:.3f}")
    print("═" * 62)

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Saved JSON report to: {json_out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriSync multimodal deepfake inference.")
    parser.add_argument("video_path", nargs="?", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.video_path is None:
        print("Usage: python deepfake_detector.py <video_path> [--json-out report.json]")
        sys.exit(1)
    detect_deepfake(args.video_path, json_out=args.json_out)
