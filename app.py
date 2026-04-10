

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title=" Deepfake Forensic System ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ───────────────────────────────────────────────
css_path = Path(__file__).parent / "streamlit_styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Lazy loading of heavy modules (keeps app startup fast)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading forensic experts…")
def load_experts():
    """Load all ML models once and cache them."""
    import tensorflow as tf
    from lip_sync_analyzer import LipSyncAnalyzer

    def _load(paths):
        for p in paths:
            if os.path.exists(p):
                try:
                    return tf.keras.models.load_model(p), p
                except Exception:
                    continue
        return None, None

    vis_model, vis_path = _load([
        "visual_expert_best.keras",
    ])
    aud_model, aud_path = _load([
        "audio_expert_best.keras", "audio_expert_best.h5",
    ])
    sync = LipSyncAnalyzer()

    # Load calibration thresholds
    cal = {
        "visual_real_threshold": 0.5,
        "audio_real_threshold": 0.5,
        "fusion_real_threshold": 0.5,
        "lip_sync_real_threshold": 0.61,
    }
    if os.path.exists("calibration.json"):
        try:
            with open("calibration.json") as f:
                cal.update(json.load(f))
        except Exception:
            pass

    return vis_model, vis_path, aud_model, aud_path, sync, cal


# ══════════════════════════════════════════════════════════════════
#  Helper — run analysis
# ══════════════════════════════════════════════════════════════════

def run_full_analysis(video_path, vis_model, aud_model, sync_analyzer, calibration):
    """Run all four expert analyses on a video and return a report dict."""
    from preprocess_utils import extract_and_crop_faces, extract_mel_spectrogram
    from frequency_analyzer import analyze_video_frequency

    # ── 1) Visual ──────────────────────────────────────────
    faces = extract_and_crop_faces(video_path, num_frames=10)
    faces_batch = np.expand_dims(faces, axis=0)
    vis_prob = 0.5
    vis_thr = float(calibration.get("visual_real_threshold", 0.5))
    if vis_model is not None:
        vis_prob = float(vis_model.predict(faces_batch, verbose=0)[0][0])

    # ── 2) Audio ───────────────────────────────────────────
    spec = extract_mel_spectrogram(video_path)
    is_silent = bool(np.all(spec == 0.5))
    aud_prob = 0.5
    aud_thr = float(calibration.get("audio_real_threshold", 0.5))
    if not is_silent and aud_model is not None:
        aud_prob = float(aud_model.predict(np.expand_dims(spec, 0), verbose=0)[0][0])

    # ── 3) Lip-sync ────────────────────────────────────────
    sync_score = 0.5
    sync_thr = float(calibration.get("lip_sync_real_threshold", 0.61))
    if not is_silent:
        try:
            s = sync_analyzer.compute_sync_score(video_path)
            if s is not None:
                sync_score = float(s)
        except Exception:
            pass

    # ── 4) Frequency ───────────────────────────────────────
    freq_summary, freq_per_frame = analyze_video_frequency(faces)
    freq_anomaly = freq_summary.get("anomaly_score", 0.5)

    # ── 5) Adaptive Fusion ─────────────────────────────────
    def _confidence(p):
        eps = 1e-7
        p_c = np.clip(p, eps, 1 - eps)
        entropy = -(p_c * np.log2(p_c) + (1 - p_c) * np.log2(1 - p_c))
        return 1.0 - entropy  # higher = more confident

    if is_silent:
        fused = 0.75 * vis_prob + 0.25 * (1.0 - freq_anomaly)
    else:
        conf_v = _confidence(vis_prob)
        conf_a = _confidence(aud_prob)
        conf_s = max(0.1, abs(sync_score - 0.5) * 2)
        conf_f = max(0.1, abs(freq_anomaly - 0.5) * 2)
        total = conf_v + conf_a + conf_s + conf_f + 1e-8
        w_v = conf_v / total
        w_a = conf_a / total
        w_s = conf_s / total
        w_f = conf_f / total
        fused = w_v * vis_prob + w_a * aud_prob + w_s * sync_score + w_f * (1.0 - freq_anomaly)

    fused_thr = float(calibration.get("fusion_real_threshold", 0.5))
    final_verdict = "REAL" if fused >= fused_thr else "FAKE"

    return {
        "faces": faces,
        "spectrogram": spec,
        "is_silent": is_silent,
        "scores": {
            "visual": vis_prob,
            "audio": aud_prob,
            "lip_sync": sync_score,
            "frequency_anomaly": freq_anomaly,
            "fused": fused,
        },
        "thresholds": {
            "visual": vis_thr,
            "audio": aud_thr,
            "lip_sync": sync_thr,
            "fusion": fused_thr,
        },
        "verdicts": {
            "visual": "REAL" if vis_prob >= vis_thr else "FAKE",
            "audio": "REAL" if aud_prob >= aud_thr else "FAKE" if not is_silent else "N/A",
            "lip_sync": "REAL" if sync_score >= sync_thr else "FAKE" if not is_silent else "N/A",
            "frequency": "CLEAN" if freq_anomaly < 0.45 else "SUSPECT",
            "final": final_verdict,
        },
        "freq_summary": freq_summary,
        "freq_per_frame": freq_per_frame,
    }


# ══════════════════════════════════════════════════════════════════
#  Plotly gauge chart
# ══════════════════════════════════════════════════════════════════

def make_gauge(value, title, is_real_prob=True):
    """Create a sleek radial gauge chart."""
    if is_real_prob:
        color = "#10b981" if value >= 0.5 else "#ef4444"
        bar_color = "#10b981" if value >= 0.5 else "#ef4444"
    else:
        color = "#ef4444" if value >= 0.5 else "#10b981"
        bar_color = "#ef4444" if value >= 0.5 else "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 36, "family": "JetBrains Mono", "color": color}},
        title={"text": title, "font": {"size": 14, "family": "Inter", "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(239,68,68,0.08)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.06)"},
                {"range": [60, 100], "color": "rgba(16,185,129,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 2},
                "thickness": 0.8,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
#  Main UI
# ══════════════════════════════════════════════════════════════════

def main():

    # ── Hero ──────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <h1 style="font-size:2.8rem;margin-bottom:0.2rem;">
                🔬 VeriSync Forensic Lab
            </h1>
            <p style="color:#94a3b8;font-size:1.1rem;margin-top:0;">
                Multi-Modal Deepfake Detection · Visual · Audio · Lip-Sync · Frequency
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Upload ────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "🎬 Upload a video for forensic analysis",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supports MP4, AVI, MOV, MKV up to 200 MB",
        )
    with col_info:
        st.markdown(
            """
            <div style="background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.15);
                 border-radius:12px;padding:1rem;margin-top:0.5rem;">
                <p style="color:#06b6d4;font-weight:600;margin:0 0 0.3rem;">How it works</p>
                <p style="color:#94a3b8;font-size:0.82rem;margin:0;line-height:1.5;">
                    VeriSync analyses your video through <b>4 independent forensic experts</b>
                    and reports exactly how manipulated the visual and audio streams are.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded is None:
        _render_landing()
        return

    # ── Save upload to temp file ──────────────────────────
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    # ── Wait for button click ─────────────────────────────
    if st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True):
        # ── Load models ───────────────────────────────────────
        vis_model, vis_path, aud_model, aud_path, sync_analyzer, calibration = load_experts()

        if vis_model is None:
            st.error("⚠️ Visual model not found. Please train the model first (`python train_visual.py`).")
        else:
            # ── Run analysis ──────────────────────────────────────
            with st.spinner("🔍 Running 4-expert forensic analysis…"):
                report = run_full_analysis(video_path, vis_model, aud_model, sync_analyzer, calibration)

            _render_results(report, uploaded.name, video_path)

            # Cleanup
            try:
                os.unlink(video_path)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════
#  Landing page (no video uploaded yet)
# ══════════════════════════════════════════════════════════════════

def _render_landing():
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("🧠", "Visual Expert", "EfficientNetV2B0 + BiGRU + Multi-Head Attention for temporal face analysis"),
        ("🎵", "Audio Expert", "Mel-spectrogram analysis with pretrained CNN to detect synthetic speech"),
        ("👄", "Lip-Sync Check", "MediaPipe landmark tracking correlated with audio energy envelope"),
        ("📊", "Frequency Analysis", "DCT & FFT spectral forensics to detect GAN-generated artifacts"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                     border-radius:16px;padding:1.5rem;text-align:center;height:220px;
                     transition:all 0.3s ease;">
                    <div style="font-size:2.5rem;margin-bottom:0.5rem;">{icon}</div>
                    <h4 style="color:#f1f5f9;font-size:1rem;margin-bottom:0.5rem;">{title}</h4>
                    <p style="color:#94a3b8;font-size:0.8rem;line-height:1.5;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
#  Results display
# ══════════════════════════════════════════════════════════════════

def _render_results(report, filename, video_path):
    scores = report["scores"]
    verdicts = report["verdicts"]

    # ── Analysis Banner ───────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center;padding:1.5rem 0;">
            <div style="font-size:1.5rem;font-weight:bold;color:#f1f5f9;">Forensic Analysis Complete</div>
            <p style="color:#94a3b8;font-size:0.9rem;margin-top:0.8rem;">
                Analysed: {filename}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Expert Score Gauges ───────────────────────────────
    st.markdown("### 📊 Forgery Detection Results")
    gauge_cols = st.columns(4)

    with gauge_cols[0]:
        fake_prob_vis = 1.0 - scores["visual"]
        st.plotly_chart(
            make_gauge(fake_prob_vis, "Video Manipulation", is_real_prob=False),
            use_container_width=True, key="gauge_vis",
        )
        verdict_color = "#10b981" if fake_prob_vis < 0.5 else "#ef4444"
        st.markdown(
            f"<p style='text-align:center;color:{verdict_color};font-weight:700;'>"
            f"{'CLEAN' if fake_prob_vis < 0.5 else 'FAKE VISUAL'}</p>",
            unsafe_allow_html=True,
        )

    with gauge_cols[1]:
        if report["is_silent"]:
            st.markdown(
                "<div style='text-align:center;padding:3rem 0;color:#64748b;'>"
                "<p style='font-size:2rem;'>🔇</p><p>Silent audio — skipped</p></div>",
                unsafe_allow_html=True,
            )
        else:
            fake_prob_aud = 1.0 - scores["audio"]
            st.plotly_chart(
                make_gauge(fake_prob_aud, "Audio Manipulation", is_real_prob=False),
                use_container_width=True, key="gauge_aud",
            )
            vc = "#10b981" if fake_prob_aud < 0.5 else "#ef4444"
            st.markdown(
                f"<p style='text-align:center;color:{vc};font-weight:700;'>"
                f"{'CLEAN' if fake_prob_aud < 0.5 else 'FAKE AUDIO'}</p>",
                unsafe_allow_html=True,
            )

    with gauge_cols[2]:
        if report["is_silent"]:
            st.markdown(
                "<div style='text-align:center;padding:3rem 0;color:#64748b;'>"
                "<p style='font-size:2rem;'>🔇</p><p>Silent — no lip-sync</p></div>",
                unsafe_allow_html=True,
            )
        else:
            fake_prob_sync = 1.0 - scores["lip_sync"]
            st.plotly_chart(
                make_gauge(fake_prob_sync, "Lip-Sync Mismatch", is_real_prob=False),
                use_container_width=True, key="gauge_sync",
            )
            vc = "#10b981" if fake_prob_sync < 0.5 else "#ef4444"
            st.markdown(
                f"<p style='text-align:center;color:{vc};font-weight:700;'>"
                f"{'MATCHED' if fake_prob_sync < 0.5 else 'MISMATCHED'}</p>",
                unsafe_allow_html=True,
            )

    with gauge_cols[3]:
        st.plotly_chart(
            make_gauge(scores["frequency_anomaly"], "Frequency Anomaly", is_real_prob=False),
            use_container_width=True, key="gauge_freq",
        )
        vc = "#10b981" if verdicts["frequency"] == "CLEAN" else "#f59e0b"
        st.markdown(
            f"<p style='text-align:center;color:{vc};font-weight:700;'>"
            f"{verdicts['frequency']}</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Tabbed Deep-Dive ──────────────────────────────────
    tab_frames, tab_gradcam, tab_freq, tab_spec, tab_report = st.tabs([
        "🎞️ Face Crops", "🔥 Grad-CAM", "📊 Frequency", "🎵 Spectrogram", "📋 Full Report",
    ])

    # ── Tab 1: Face crops ─────────────────────────────────
    with tab_frames:
        st.markdown("#### Extracted Face Crops (10 Frames)")
        face_cols = st.columns(5)
        for i, col in enumerate(face_cols):
            with col:
                if i < len(report["faces"]):
                    frame = report["faces"][i]
                    if frame.max() > 1:
                        frame = frame.astype(np.uint8)
                    st.image(frame, caption=f"Frame {i+1}", use_container_width=True)
        face_cols2 = st.columns(5)
        for i, col in enumerate(face_cols2):
            with col:
                idx = i + 5
                if idx < len(report["faces"]):
                    frame = report["faces"][idx]
                    if frame.max() > 1:
                        frame = frame.astype(np.uint8)
                    st.image(frame, caption=f"Frame {idx+1}", use_container_width=True)

    # ── Tab 2: Grad-CAM ──────────────────────────────────
    with tab_gradcam:
        st.markdown("#### 🔥 Grad-CAM Attention Heatmaps")
        st.caption("Highlighted regions show where the model focuses to detect manipulation")

        try:
            from gradcam import generate_gradcam_gallery
            vis_model_ref = load_experts()[0]
            if vis_model_ref is not None:
                faces_batch = np.expand_dims(report["faces"], axis=0)
                gallery = generate_gradcam_gallery(vis_model_ref, faces_batch)

                for row_start in range(0, min(len(gallery), 10), 5):
                    gcam_cols = st.columns(5)
                    for j, col in enumerate(gcam_cols):
                        idx = row_start + j
                        if idx < len(gallery):
                            orig, overlay, hm = gallery[idx]
                            with col:
                                if orig.max() > 1:
                                    orig = orig.astype(np.uint8)
                                overlay = overlay.astype(np.uint8)
                                st.image(overlay, caption=f"Frame {idx+1}", use_container_width=True)
            else:
                st.warning("Visual model not loaded — cannot generate Grad-CAM.")
        except Exception as e:
            st.warning(f"Grad-CAM generation failed: {e}")

    # ── Tab 3: Frequency analysis ─────────────────────────
    with tab_freq:
        st.markdown("#### 📊 Spectral Forensics")

        freq_col1, freq_col2 = st.columns(2)
        with freq_col1:
            st.markdown("**Per-Frame Anomaly Scores**")
            per_frame = report["freq_per_frame"]
            fig_freq = go.Figure()
            colors = ["#ef4444" if s > 0.45 else "#10b981" for s in per_frame]
            fig_freq.add_trace(go.Bar(
                x=[f"F{i+1}" for i in range(len(per_frame))],
                y=per_frame,
                marker_color=colors,
                text=[f"{s:.2f}" for s in per_frame],
                textposition="outside",
                textfont={"family": "JetBrains Mono", "size": 11},
            ))
            fig_freq.add_hline(y=0.45, line_dash="dash", line_color="#f59e0b",
                              annotation_text="Threshold", annotation_position="top left")
            fig_freq.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#94a3b8"},
                xaxis={"title": "Frame", "gridcolor": "rgba(255,255,255,0.05)"},
                yaxis={"title": "Anomaly Score", "range": [0, 1],
                       "gridcolor": "rgba(255,255,255,0.05)"},
                height=350, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_freq, use_container_width=True)

        with freq_col2:
            st.markdown("**Frequency Spectrum (Sample Frame)**")
            try:
                from frequency_analyzer import generate_frequency_spectrum_image
                spec_img = generate_frequency_spectrum_image(report["faces"][0])
                st.image(spec_img, caption="2D FFT Magnitude Spectrum", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate spectrum: {e}")

            # Feature table
            fs = report["freq_summary"]
            st.markdown("**Feature Summary**")
            feat_data = {
                "Feature": ["High-Freq Ratio", "Spectral Flatness", "DCT Kurtosis",
                           "Roll-off Slope", "Spectral Entropy"],
                "Value": [
                    f"{fs.get('high_freq_ratio', 0):.4f}",
                    f"{fs.get('spectral_flatness', 0):.4f}",
                    f"{fs.get('dct_kurtosis', 0):.2f}",
                    f"{fs.get('rolloff_slope', 0):.3f}",
                    f"{fs.get('spectral_entropy', 0):.2f}",
                ],
            }
            st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    # ── Tab 4: Spectrogram ────────────────────────────────
    with tab_spec:
        st.markdown("#### 🎵 Audio Mel-Spectrogram")
        if report["is_silent"]:
            st.info("Audio is silent — no spectrogram to display.")
        else:
            spec_display = report["spectrogram"][:, :, 0]
            fig_spec = go.Figure(go.Heatmap(
                z=np.flipud(spec_display),
                colorscale="Inferno",
                showscale=True,
                colorbar={"title": {"text": "dB", "font": {"color": "#94a3b8"}},
                          "tickfont": {"color": "#94a3b8"}},
            ))
            fig_spec.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#94a3b8"},
                xaxis={"title": "Time Frame"},
                yaxis={"title": "Mel Frequency Bin"},
                height=350, margin=dict(l=60, r=20, t=20, b=50),
            )
            st.plotly_chart(fig_spec, use_container_width=True)

    # ── Tab 5: Full JSON Report ───────────────────────────
    with tab_report:
        st.markdown("#### 📋 Forensic Evidence Report")

        export_report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "filename": filename if 'filename' in dir() else "unknown",
            "scores": report["scores"],
            "thresholds": report["thresholds"],
            "verdicts": report["verdicts"],
            "frequency_features": report["freq_summary"],
            "flags": {"is_silent_audio": report["is_silent"]},
        }

        st.json(export_report)

        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(export_report, indent=2),
            file_name=f"verisync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
