import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def _ensure_columns(df):
    out = df.copy()

    if "video_label" not in out.columns:
        out["video_label"] = out["category"].apply(
            lambda x: 1 if str(x).startswith("RealVideo") or str(x) == "original" else 0
        )

    if "audio_label" not in out.columns:
        out["audio_label"] = out.apply(
            lambda r: 1
            if bool(r.get("has_audio", False)) and "RealAudio" in str(r.get("category", ""))
            else (0 if bool(r.get("has_audio", False)) else -1),
            axis=1,
        )

    if "subject_id" not in out.columns:
        out["subject_id"] = out["video_path"].astype(str).str.extract(r"(id\d+)")[0].fillna("unknown_subject")

    if "source_id" not in out.columns:
        out["source_id"] = out["video_path"].map(lambda p: f"src::{Path(p).stem}")

    return out


def _stratified_group_split(df, label_col, group_col, val_ratio, seed):
    n_splits = max(2, round(1.0 / val_ratio))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, df[label_col], groups=df[group_col]))
        return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    except Exception:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
        return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _cap_fake_ratio(train_df, label_col, fake_to_real_ratio, seed):
    if label_col not in train_df:
        return train_df

    real_df = train_df[train_df[label_col] == 1]
    fake_df = train_df[train_df[label_col] == 0]

    if len(real_df) == 0 or len(fake_df) == 0:
        return train_df

    max_fake = int(len(real_df) * fake_to_real_ratio)
    if len(fake_df) <= max_fake:
        return train_df

    fake_df = fake_df.sample(n=max_fake, random_state=seed)
    return pd.concat([real_df, fake_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _print_overlap_stats(name, train_df, val_df, label_col, group_col):
    train_paths = set(train_df["video_path"])
    val_paths = set(val_df["video_path"])
    path_overlap = len(train_paths & val_paths)

    train_groups = set(train_df[group_col].astype(str))
    val_groups = set(val_df[group_col].astype(str))
    group_overlap = len(train_groups & val_groups)

    print(f"\n{name} split summary")
    print("-" * (len(name) + 14))
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Train labels: {train_df[label_col].value_counts().to_dict()}")
    print(f"Val labels:   {val_df[label_col].value_counts().to_dict()}")
    print(f"Path overlap:  {path_overlap}")
    print(f"Group overlap: {group_overlap}")


def create_splits(
    manifest_path="data_manifest.csv",
    seed=42,
    visual_val_ratio=0.2,
    audio_val_ratio=0.15,
    visual_fake_to_real_ratio=3.0,
    visual_train_out="train_manifest.csv",
    visual_val_out="val_manifest.csv",
    audio_train_out="train_audio_manifest.csv",
    audio_val_out="val_audio_manifest.csv",
):
    df = pd.read_csv(manifest_path)
    df = _ensure_columns(df)

    # Visual task uses video_label only.
    visual_df = df[df["video_label"].isin([0, 1])].copy()
    visual_df["label"] = visual_df["video_label"].astype(int)
    visual_df["split_group"] = (
        visual_df["source_id"].fillna(visual_df["subject_id"]).fillna(visual_df["video_path"].map(lambda p: Path(p).stem))
    )

    visual_train, visual_val = _stratified_group_split(
        visual_df,
        label_col="label",
        group_col="split_group",
        val_ratio=visual_val_ratio,
        seed=seed,
    )
    visual_train = _cap_fake_ratio(
        visual_train,
        label_col="label",
        fake_to_real_ratio=visual_fake_to_real_ratio,
        seed=seed,
    )

    # Audio task uses audio_label only and only samples with audio.
    audio_df = df[(df["has_audio"] == True) & (df["audio_label"].isin([0, 1]))].copy()
    audio_df["label"] = audio_df["audio_label"].astype(int)
    audio_df["split_group"] = (
        audio_df["subject_id"].fillna(audio_df["source_id"]).fillna(audio_df["video_path"].map(lambda p: Path(p).stem))
    )

    audio_train, audio_val = _stratified_group_split(
        audio_df,
        label_col="label",
        group_col="split_group",
        val_ratio=audio_val_ratio,
        seed=seed,
    )

    visual_train.to_csv(visual_train_out, index=False)
    visual_val.to_csv(visual_val_out, index=False)
    audio_train.to_csv(audio_train_out, index=False)
    audio_val.to_csv(audio_val_out, index=False)

    _print_overlap_stats("Visual", visual_train, visual_val, label_col="label", group_col="split_group")
    _print_overlap_stats("Audio", audio_train, audio_val, label_col="label", group_col="split_group")

    print("\nSaved split files:")
    print(f"- {visual_train_out}")
    print(f"- {visual_val_out}")
    print(f"- {audio_train_out}")
    print(f"- {audio_val_out}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare leakage-safe splits for visual/audio deepfake tasks.")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-val-ratio", type=float, default=0.2)
    parser.add_argument("--audio-val-ratio", type=float, default=0.15)
    parser.add_argument("--visual-fake-to-real-ratio", type=float, default=3.0)
    parser.add_argument("--visual-train-out", default="train_manifest.csv")
    parser.add_argument("--visual-val-out", default="val_manifest.csv")
    parser.add_argument("--audio-train-out", default="train_audio_manifest.csv")
    parser.add_argument("--audio-val-out", default="val_audio_manifest.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_splits(
        manifest_path=args.manifest,
        seed=args.seed,
        visual_val_ratio=args.visual_val_ratio,
        audio_val_ratio=args.audio_val_ratio,
        visual_fake_to_real_ratio=args.visual_fake_to_real_ratio,
        visual_train_out=args.visual_train_out,
        visual_val_out=args.visual_val_out,
        audio_train_out=args.audio_train_out,
        audio_val_out=args.audio_val_out,
    )
