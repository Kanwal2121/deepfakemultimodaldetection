"""
Frequency Domain Deepfake Analyzer — 4th Modality for VeriSync.

GANs and face-swapping pipelines leave characteristic spectral fingerprints
that are invisible in pixel space but clearly visible in the frequency domain.
This module uses DCT and FFT analysis to detect such artifacts without any
trained neural network — purely signal-processing based.

Key techniques:
  1. 2D DCT energy distribution — deepfakes often have suppressed high-freq.
  2. Azimuthal FFT averaging — GANs produce characteristic radial patterns.
  3. Statistical features: kurtosis, spectral entropy, band energy ratios.
"""

import cv2
import numpy as np
from scipy import fftpack, stats


# ──────────────────────────────────────────────────────────────────────
#  Core analysis functions
# ──────────────────────────────────────────────────────────────────────

def _to_grayscale(image):
    """Convert (H, W, 3) float/uint8 image to (H, W) float64 grayscale."""
    if image.ndim == 3:
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return gray.astype(np.float64)


def compute_dct_features(gray, precomputed_dct=None):
    """
    Compute energy distribution features from the 2D DCT of a grayscale image.

    Returns a dict with:
        - high_freq_ratio: fraction of energy in high-frequency DCT bins
        - mid_freq_ratio:  fraction of energy in mid-frequency bins
        - low_freq_ratio:  fraction of energy in low-frequency bins
        - spectral_entropy: Shannon entropy of normalized DCT energy
    Also returns the computed DCT for reuse by other functions.
    """
    dct = precomputed_dct if precomputed_dct is not None else fftpack.dct(fftpack.dct(gray.T, norm="ortho").T, norm="ortho")
    magnitude = np.abs(dct)
    energy = magnitude ** 2

    h, w = gray.shape
    total_energy = energy.sum() + 1e-12

    # Define frequency bands by distance from DC component (top-left)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy / h) ** 2 + (xx / w) ** 2)

    low_mask = dist < 0.15
    mid_mask = (dist >= 0.15) & (dist < 0.5)
    high_mask = dist >= 0.5

    low_energy = energy[low_mask].sum() / total_energy
    mid_energy = energy[mid_mask].sum() / total_energy
    high_energy = energy[high_mask].sum() / total_energy

    # Spectral entropy
    flat = (energy / total_energy).flatten()
    flat = flat[flat > 0]
    entropy = -np.sum(flat * np.log2(flat + 1e-20))

    features = {
        "low_freq_ratio": float(low_energy),
        "mid_freq_ratio": float(mid_energy),
        "high_freq_ratio": float(high_energy),
        "spectral_entropy": float(entropy),
    }
    return features, dct


def compute_fft_azimuthal(gray):
    """
    Compute the azimuthally averaged 1D power spectrum from the 2D FFT.

    GANs produce characteristic patterns in the azimuthal average that differ
    from natural images — particularly periodic peaks or unnaturally smooth
    roll-off curves.

    Returns:
        radial_profile: 1D np.array of averaged power at each radial frequency
        spectral_flatness: ratio of geometric to arithmetic mean (1 = white noise)
        rolloff_slope: linear regression slope of log-power vs log-frequency
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)

    # Azimuthal average
    yy, xx = np.mgrid[0:h, 0:w]
    radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)

    radial_profile = np.zeros(max_radius)
    for r in range(max_radius):
        mask = radii == r
        if mask.any():
            radial_profile[r] = magnitude[mask].mean()

    # Skip DC component
    profile = radial_profile[1:]
    profile = np.maximum(profile, 1e-12)

    # Spectral flatness
    log_profile = np.log(profile)
    geometric_mean = np.exp(np.mean(log_profile))
    arithmetic_mean = np.mean(profile)
    spectral_flatness = float(geometric_mean / (arithmetic_mean + 1e-12))

    # Roll-off slope (linear fit on log-log scale)
    freqs = np.arange(1, len(profile) + 1)
    log_freqs = np.log(freqs)
    log_power = np.log(profile)
    slope, _, _, _, _ = stats.linregress(log_freqs, log_power)

    return radial_profile, float(spectral_flatness), float(slope)


def compute_kurtosis_features(gray, precomputed_dct=None):
    """
    Kurtosis of DCT coefficients — real images tend to have higher kurtosis
    (heavier tails) while GAN-generated images have more Gaussian distributions.
    """
    dct = precomputed_dct if precomputed_dct is not None else fftpack.dct(fftpack.dct(gray.T, norm="ortho").T, norm="ortho")
    flat = dct.flatten()
    k = float(stats.kurtosis(flat, fisher=True))
    return {"dct_kurtosis": k}


# ──────────────────────────────────────────────────────────────────────
#  Main analysis pipeline
# ──────────────────────────────────────────────────────────────────────

def analyze_frame_frequency(frame):
    """
    Run full frequency analysis on a single frame.

    Parameters
    ----------
    frame : np.ndarray
        (H, W, 3) image in RGB, float32 [0, 255] or uint8.

    Returns
    -------
    features : dict
        All frequency domain features.
    anomaly_score : float
        Combined anomaly score ∈ [0, 1], higher = more likely fake.
    """
    gray = _to_grayscale(frame)

    # Compute DCT once, reuse across feature extractors
    dct_feats, dct_matrix = compute_dct_features(gray)
    radial_profile, spectral_flatness, rolloff_slope = compute_fft_azimuthal(gray)
    kurt_feats = compute_kurtosis_features(gray, precomputed_dct=dct_matrix)

    features = {
        **dct_feats,
        "spectral_flatness": spectral_flatness,
        "rolloff_slope": rolloff_slope,
        **kurt_feats,
    }

    # ── Refined anomaly score (Continuous) ──
    # Instead of hard thresholds, we map features to a [0, 1] anomaly range
    # based on typical distributions for real (clean) vs fake (GAN) faces.
    
    def _map_anomaly(val, clean_ref, fake_ref):
        """Map value to [0, 1] where 1.0 is maximally anomalous."""
        if abs(clean_ref - fake_ref) < 1e-7: return 0.5
        if clean_ref < fake_ref: # Higher is more fake (e.g. flatness)
            score = (val - clean_ref) / (fake_ref - clean_ref)
        else: # Lower is more fake (e.g. high-freq ratio, kurtosis)
            score = (clean_ref - val) / (clean_ref - fake_ref)
        return float(np.clip(score, 0.0, 1.0))

    # Features: Clean Center, Fake Center (empirically derived)
    s_hf = _map_anomaly(dct_feats["high_freq_ratio"], 0.08, 0.02)
    s_flat = _map_anomaly(spectral_flatness, 0.05, 0.18)
    s_kurt = _map_anomaly(kurt_feats["dct_kurtosis"], 15.0, 3.0)
    s_slope = _map_anomaly(rolloff_slope, -3.2, -1.8)
    s_ent = _map_anomaly(dct_feats["spectral_entropy"], 8.8, 11.2)

    # Combined score (weighted: high-freq and kurtosis are strongest signals)
    anomaly_score = (s_hf*0.3 + s_kurt*0.3 + s_flat*0.15 + s_slope*0.15 + s_ent*0.1)
    anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))
    
    features["anomaly_score"] = anomaly_score


    return features, anomaly_score


def analyze_video_frequency(frames, aggregate="median"):
    """
    Analyze frequency features across multiple frames of a video.

    Parameters
    ----------
    frames : np.ndarray
        (T, H, W, 3) face crops.
    aggregate : str
        How to combine per-frame scores: "mean", "median", or "max".

    Returns
    -------
    summary : dict
        Aggregated features and final anomaly score.
    per_frame_scores : list[float]
        Individual frame anomaly scores.
    """
    frame_scores = []
    all_features = []

    for t in range(len(frames)):
        feats, score = analyze_frame_frequency(frames[t])
        frame_scores.append(score)
        all_features.append(feats)

    if aggregate == "mean":
        final_score = float(np.mean(frame_scores))
    elif aggregate == "max":
        final_score = float(np.max(frame_scores))
    else:  # median
        final_score = float(np.median(frame_scores))

    # Average features across frames
    summary = {}
    for key in all_features[0]:
        vals = [f[key] for f in all_features]
        summary[key] = float(np.mean(vals))
    summary["anomaly_score"] = final_score
    summary["per_frame_std"] = float(np.std(frame_scores))

    return summary, frame_scores


# ──────────────────────────────────────────────────────────────────────
#  Visualization helper for the dashboard
# ──────────────────────────────────────────────────────────────────────

def generate_frequency_spectrum_image(frame, size=(256, 256)):
    """
    Generate a visual representation of the 2D FFT magnitude spectrum.
    Returns an RGB image suitable for display in the Streamlit dashboard.
    """
    gray = _to_grayscale(frame)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Normalize to [0, 255]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max - mag_min > 0:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min) * 255
    magnitude = magnitude.astype(np.uint8)
    magnitude = cv2.resize(magnitude, size)

    # Apply colormap for visual appeal
    colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


# ──────────────────────────────────────────────────────────────────────
#  CLI test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import pandas as pd
    from preprocess_utils import extract_and_crop_faces

    if not os.path.exists("data_manifest.csv"):
        print("data_manifest.csv not found")
        exit(1)

    df = pd.read_csv("data_manifest.csv")
    real_video = df[df["label"] == 1].iloc[0]["video_path"]
    fake_video = df[df["label"] == 0].iloc[0]["video_path"]

    for path, name in [(real_video, "REAL"), (fake_video, "FAKE")]:
        print(f"\n{'='*50}")
        print(f"Analyzing {name}: {os.path.basename(path)}")
        print("=" * 50)
        frames = extract_and_crop_faces(path, num_frames=5)
        summary, per_frame = analyze_video_frequency(frames)
        print(f"  Anomaly Score: {summary['anomaly_score']:.3f}")
        print(f"  High-Freq Ratio: {summary['high_freq_ratio']:.4f}")
        print(f"  Spectral Flatness: {summary['spectral_flatness']:.4f}")
        print(f"  DCT Kurtosis: {summary['dct_kurtosis']:.2f}")
        print(f"  Roll-off Slope: {summary['rolloff_slope']:.3f}")
        print(f"  Per-frame scores: {[f'{s:.2f}' for s in per_frame]}")