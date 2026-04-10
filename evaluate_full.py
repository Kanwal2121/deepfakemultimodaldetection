"""
VeriSync — Full Pipeline Evaluator
═══════════════════════════════════
Runs all experts on the validation set and generates publication-quality
evaluation plots: ROC curve, PR curve, confusion matrix, and per-manipulation
performance breakdown.

Usage:
    python evaluate_full.py
    python evaluate_full.py --visual-manifest val_manifest.csv --output-dir eval_results
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from data_loader import create_visual_dataset
from data_loader_audio import create_audio_dataset


# ── Plotting style ────────────────────────────────────────────────
COLORS = {
    "primary": "#06b6d4",
    "secondary": "#8b5cf6",
    "accent": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "bg": "#0a0e1a",
    "card": "#111827",
    "text": "#f1f5f9",
    "muted": "#64748b",
}


def _setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["card"],
        "axes.edgecolor": COLORS["muted"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "grid.color": "#1e293b",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def _save_fig(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  Plot generators
# ══════════════════════════════════════════════════════════════════

def plot_roc_curve(y_true, y_score, title, out_path):
    _setup_plot_style()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])
    ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], linewidth=1, alpha=0.7)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_path)
    return roc_auc


def plot_pr_curve(y_true, y_score, title, out_path):
    _setup_plot_style()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS["secondary"])
    ax.plot(recall, precision, color=COLORS["secondary"], linewidth=2.5,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_path)
    return ap


def plot_confusion_matrix(y_true, y_pred, title, out_path, labels=("Fake", "Real")):
    _setup_plot_style()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="BuPu")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > thresh else COLORS["text"])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_path)


def plot_per_manipulation(df_results, out_path):
    """Bar chart showing accuracy per manipulation type."""
    _setup_plot_style()

    if "manipulation_type" not in df_results.columns or "correct" not in df_results.columns:
        return

    grouped = df_results.groupby("manipulation_type")["correct"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(grouped) * 0.6)))
    bar_colors = [COLORS["primary"] if v >= 0.7 else COLORS["warning"] if v >= 0.5
                  else COLORS["danger"] for v in grouped.values]
    bars = ax.barh(range(len(grouped)), grouped.values, color=bar_colors, height=0.6,
                   edgecolor="none", alpha=0.9)
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped.index, fontsize=10)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Manipulation Type", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim([0, 1])
    ax.axvline(x=0.5, color=COLORS["danger"], linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, grouped.values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10, fontweight="600")

    _save_fig(fig, out_path)


def plot_score_distribution(y_true, y_score, title, out_path):
    """Overlapping histograms of prediction scores for real vs. fake."""
    _setup_plot_style()

    real_scores = y_score[y_true == 1]
    fake_scores = y_score[y_true == 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(real_scores, bins=40, alpha=0.6, color=COLORS["accent"], label="Real", density=True)
    ax.hist(fake_scores, bins=40, alpha=0.6, color=COLORS["danger"], label="Fake", density=True)
    ax.set_xlabel("P(Real) Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_path)


# ══════════════════════════════════════════════════════════════════
#  Main evaluation
# ══════════════════════════════════════════════════════════════════

def _load_model(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                return tf.keras.models.load_model(p), p
            except Exception:
                continue
    raise FileNotFoundError(f"Model not found: {paths}")


def _find_best_threshold(y_true, y_score):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation with visualizations.")
    parser.add_argument("--visual-manifest", default="val_manifest.csv")
    parser.add_argument("--audio-manifest", default="val_audio_manifest.csv")
    parser.add_argument("--visual-batch-size", type=int, default=8)
    parser.add_argument("--audio-batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("🔬 VeriSync — Full Pipeline Evaluation")
    print("=" * 60)

    results = {}

    # ── Visual Expert ─────────────────────────────────────
    if os.path.exists(args.visual_manifest):
        print("\n📸 Evaluating Visual Expert...")
        vis_model, vis_path = _load_model([
            "visual_expert_best.keras", "visual_expert_best.h5", "visual_expert_besttt.h5",
        ])
        vis_df = pd.read_csv(args.visual_manifest)
        if "video_label" in vis_df.columns:
            vis_df["label"] = vis_df["video_label"].astype(int)
        y_true_vis = vis_df["label"].values

        vis_ds = create_visual_dataset(
            args.visual_manifest, batch_size=args.visual_batch_size, shuffle=False, augment=False,
        )
        y_score_vis = vis_model.predict(vis_ds, verbose=1).reshape(-1)

        best_t, best_f1 = _find_best_threshold(y_true_vis, y_score_vis)
        y_pred_vis = (y_score_vis >= best_t).astype(int)

        vis_auc = plot_roc_curve(y_true_vis, y_score_vis, "Visual Expert — ROC Curve",
                                 out_dir / "visual_roc.png")
        vis_ap = plot_pr_curve(y_true_vis, y_score_vis, "Visual Expert — Precision-Recall",
                               out_dir / "visual_pr.png")
        plot_confusion_matrix(y_true_vis, y_pred_vis, "Visual Expert — Confusion Matrix",
                              out_dir / "visual_cm.png")
        plot_score_distribution(y_true_vis, y_score_vis, "Visual Expert — Score Distribution",
                                out_dir / "visual_scores.png")

        # Per-manipulation breakdown
        if "manipulation_type" in vis_df.columns:
            vis_df["pred"] = y_pred_vis[:len(vis_df)]
            vis_df["correct"] = (vis_df["pred"] == vis_df["label"]).astype(int)
            plot_per_manipulation(vis_df, out_dir / "visual_per_manipulation.png")

        bal_acc = balanced_accuracy_score(y_true_vis, y_pred_vis)
        results["visual"] = {
            "auroc": float(vis_auc), "auprc": float(vis_ap),
            "best_f1": float(best_f1), "best_threshold": best_t,
            "balanced_accuracy": float(bal_acc), "model_path": vis_path,
        }
        print(f"  AUROC: {vis_auc:.4f} | AUPRC: {vis_ap:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.3f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")

    # ── Audio Expert ──────────────────────────────────────
    if os.path.exists(args.audio_manifest):
        print("\n🎵 Evaluating Audio Expert...")
        aud_model, aud_path = _load_model([
            "audio_expert_best.keras", "audio_expert_best.h5",
        ])
        aud_df = pd.read_csv(args.audio_manifest)
        if "audio_label" in aud_df.columns:
            aud_df["label"] = aud_df["audio_label"].astype(int)
        y_true_aud = aud_df["label"].values

        aud_ds = create_audio_dataset(
            args.audio_manifest, batch_size=args.audio_batch_size, shuffle=False, augment=False,
        )
        y_score_aud = aud_model.predict(aud_ds, verbose=1).reshape(-1)

        best_t_a, best_f1_a = _find_best_threshold(y_true_aud, y_score_aud)
        y_pred_aud = (y_score_aud >= best_t_a).astype(int)

        aud_auc = plot_roc_curve(y_true_aud, y_score_aud, "Audio Expert — ROC Curve",
                                  out_dir / "audio_roc.png")
        aud_ap = plot_pr_curve(y_true_aud, y_score_aud, "Audio Expert — Precision-Recall",
                                out_dir / "audio_pr.png")
        plot_confusion_matrix(y_true_aud, y_pred_aud, "Audio Expert — Confusion Matrix",
                              out_dir / "audio_cm.png")
        plot_score_distribution(y_true_aud, y_score_aud, "Audio Expert — Score Distribution",
                                out_dir / "audio_scores.png")

        bal_acc_a = balanced_accuracy_score(y_true_aud, y_pred_aud)
        results["audio"] = {
            "auroc": float(aud_auc), "auprc": float(aud_ap),
            "best_f1": float(best_f1_a), "best_threshold": best_t_a,
            "balanced_accuracy": float(bal_acc_a), "model_path": aud_path,
        }
        print(f"  AUROC: {aud_auc:.4f} | AUPRC: {aud_ap:.4f} | Best F1: {best_f1_a:.4f} @ t={best_t_a:.3f}")
        print(f"  Balanced Accuracy: {bal_acc_a:.4f}")

    # ── Save results ──────────────────────────────────────
    results["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"📊 All plots and metrics saved to: {out_dir}/")
    print("=" * 60)

    # ── Console Summary Table ─────────────────────────────
    print("\n┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ Expert      │  AUROC   │  AUPRC   │  Best F1 │  Bal.Acc │")
    print("├─────────────┼──────────┼──────────┼──────────┼──────────┤")
    for name, m in results.items():
        if isinstance(m, dict) and "auroc" in m:
            print(f"│ {name:<11} │ {m['auroc']:.4f}   │ {m['auprc']:.4f}   │ {m['best_f1']:.4f}   │ {m['balanced_accuracy']:.4f}   │")
    print("└─────────────┴──────────┴──────────┴──────────┴──────────┘")


if __name__ == "__main__":
    main()
