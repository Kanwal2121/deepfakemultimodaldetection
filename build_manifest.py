"""
VeriSync — Production Data Manifest Builder
═══════════════════════════════════════════
Builds a unified manifest from FaceForensics++ C23 and FakeAVCeleb datasets
with proper subject-level IDs for leakage-safe splitting.

Usage:
    python build_manifest.py
"""

import os
import re
from pathlib import Path

import pandas as pd


def find_video_files(root_dir, extensions=(".mp4", ".avi", ".mov", ".mkv")):
    """Recursively find all video files in a directory tree."""
    video_files = []
    if not os.path.exists(root_dir):
        return []
    for dirpath, _, filenames in os.walk(root_dir):
        if ".AppleDouble" in dirpath:
            continue
        for filename in filenames:
            if filename.lower().endswith(extensions) and not filename.startswith("."):
                video_files.append(os.path.join(dirpath, filename))
    return video_files


# ══════════════════════════════════════════════════════════════════
#  Subject / Source ID extraction
# ══════════════════════════════════════════════════════════════════

def _extract_fakeavceleb_subject_id(video_path):
    """Extract subject identity (e.g., id00166) from FakeAVCeleb paths."""
    match = re.search(r"id\d+", video_path)
    return match.group(0) if match else "unknown_subject"


def _extract_ffpp_ids(video_path):
    """Extract source identity from FF++ filenames (e.g., 000_003.mp4 → source 000)."""
    stem = Path(video_path).stem
    # Fake videos: 000_003 → source=000, target=003
    pair_match = re.match(r"^(\d{3})_(\d{3})$", stem)
    if pair_match:
        a, b = pair_match.groups()
        return f"ffpp_src_{a}", f"ffpp_pair_{a}_{b}"
    # DeepFakeDetection: 02_15__walking_and_outside_surprised__MZWH8ATN
    dfd_match = re.match(r"^(\d{2})_(\d{2})__", stem)
    if dfd_match:
        a, b = dfd_match.groups()
        return f"ffpp_dfd_{a}", f"ffpp_dfd_{a}_{b}"
    # Original videos: 000 → source=000
    clip_match = re.match(r"^(\d{3})$", stem)
    if clip_match:
        a = clip_match.group(1)
        return f"ffpp_src_{a}", f"ffpp_clip_{a}"
    return "ffpp_unknown_subject", f"ffpp_unknown_{stem}"


# ══════════════════════════════════════════════════════════════════
#  Label inference
# ══════════════════════════════════════════════════════════════════

def _infer_video_label(dataset_name, category):
    """Is the VIDEO face real (1) or manipulated (0)?"""
    if dataset_name == "FakeAVCeleb":
        return 1 if category.startswith("RealVideo") else 0
    # FF++: only 'original' is real
    return 1 if category == "original" else 0


def _infer_audio_label(has_audio, category):
    """Is the AUDIO real (1), fake (0), or absent (-1)?"""
    if not has_audio:
        return -1
    return 1 if "RealAudio" in str(category) else 0


def _make_entry(video_path, dataset_name, category, has_audio, multimodal_label, manipulation_method="unknown"):
    video_label = _infer_video_label(dataset_name, category)
    audio_label = _infer_audio_label(has_audio, category)

    if dataset_name == "FakeAVCeleb":
        subject_id = _extract_fakeavceleb_subject_id(video_path)
        clip_id = Path(video_path).stem
        source_id = f"favc::{subject_id}::{clip_id}"
    else:
        subject_id, source_id = _extract_ffpp_ids(video_path)

    return {
        "video_path": video_path,
        "label": int(multimodal_label),
        "video_label": int(video_label),
        "audio_label": int(audio_label),
        "dataset": dataset_name,
        "category": category,
        "has_audio": bool(has_audio),
        "subject_id": subject_id,
        "source_id": source_id,
        "manipulation_type": manipulation_method,
    }


def create_data_manifest(output_file="data_manifest.csv"):
    """Walk both datasets and create a unified manifest."""
    data_entries = []

    # ── 1. FAKEACELEB ─────────────────────────────────────
    fav_root = "FakeAVCeleb"
    if os.path.exists(fav_root):
        fav_categories = {
            "RealVideo-RealAudio": (1, "real"),
            "RealVideo-FakeAudio": (0, "SV2TTS"),
            "FakeVideo-RealAudio": (0, "faceswap/FSGAN/Wav2Lip"),
            "FakeVideo-FakeAudio": (0, "faceswap+SV2TTS"),
        }

        # Try to use metadata for more accurate manipulation method
        meta_path = os.path.join(fav_root, "meta_data.csv")
        meta_lookup = {}
        if os.path.exists(meta_path):
            try:
                meta_df = pd.read_csv(meta_path)
                for _, row in meta_df.iterrows():
                    fname = str(row.get("filename", ""))
                    method = str(row.get("method", "unknown"))
                    path_field = str(row.get("path", ""))
                    key = os.path.join(fav_root, path_field, fname) if path_field else fname
                    meta_lookup[fname] = method
                print(f"  Loaded {len(meta_lookup)} entries from FakeAVCeleb metadata")
            except Exception as e:
                print(f"  Warning: could not parse FakeAVCeleb metadata: {e}")

        for cat, (label, default_method) in fav_categories.items():
            cat_path = os.path.join(fav_root, cat)
            video_paths = find_video_files(cat_path)
            for path in video_paths:
                fname = os.path.basename(path)
                method = meta_lookup.get(fname, default_method)
                data_entries.append(
                    _make_entry(
                        video_path=path,
                        dataset_name="FakeAVCeleb",
                        category=cat,
                        has_audio=True,
                        multimodal_label=label,
                        manipulation_method=method if label == 0 else "real",
                    )
                )
            print(f"FakeAVCeleb/{cat}: {len(video_paths)} videos")

    # ── 2. FACEFORENSICS++ C23 ────────────────────────────
    ff_root = "FaceForensics++_C23"
    if os.path.exists(ff_root):
        ff_subfolders = sorted([
            f for f in os.listdir(ff_root)
            if os.path.isdir(os.path.join(ff_root, f)) and f != "csv"
        ])
        for sub in ff_subfolders:
            sub_path = os.path.join(ff_root, sub)
            label = 1 if sub == "original" else 0
            video_paths = find_video_files(sub_path)
            for path in video_paths:
                data_entries.append(
                    _make_entry(
                        video_path=path,
                        dataset_name="FaceForensics++",
                        category=sub,
                        has_audio=False,
                        multimodal_label=label,
                        manipulation_method=sub if label == 0 else "real",
                    )
                )
            print(f"FaceForensics++/{sub}: {len(video_paths)} videos")

    if not data_entries:
        print("\n❌ No videos found! Check your dataset folders.")
        return

    df = pd.DataFrame(data_entries)
    df.to_csv(output_file, index=False)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("✅ DATA MANIFEST BUILT SUCCESSFULLY")
    print("═" * 60)
    print(f"📂 Output: {output_file}")
    print(f"📊 Total videos: {len(df):,}")
    print()

    print("── Visual Task (video_label) ──")
    vl = df["video_label"].value_counts().to_dict()
    real_v = vl.get(1, 0)
    fake_v = vl.get(0, 0)
    print(f"   Real: {real_v:>6,}  |  Fake: {fake_v:>6,}  |  Ratio: 1:{fake_v/max(real_v,1):.1f}")

    print("── Audio Task (audio_label) ──")
    al = df[df["audio_label"] >= 0]["audio_label"].value_counts().to_dict()
    real_a = al.get(1, 0)
    fake_a = al.get(0, 0)
    print(f"   Real: {real_a:>6,}  |  Fake: {fake_a:>6,}  |  Ratio: 1:{fake_a/max(real_a,1):.1f}")
    print(f"   No audio: {len(df[df['audio_label'] == -1]):,}")

    print("\n── By Manipulation Type ──")
    for mt, count in df["manipulation_type"].value_counts().items():
        lbl = "REAL" if mt == "real" else "FAKE"
        print(f"   {mt:<30} {count:>6,}  [{lbl}]")

    print("\n── By Dataset ──")
    for ds, count in df["dataset"].value_counts().items():
        print(f"   {ds:<20} {count:>6,}")

    print("═" * 60)


if __name__ == "__main__":
    create_data_manifest()
