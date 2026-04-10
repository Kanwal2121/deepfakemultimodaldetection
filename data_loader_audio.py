# data_loader_audio.py — Production Audio Data Pipeline
# Enhanced with: SpecAugment (time/frequency masking), noise injection, time shift

import pandas as pd
import tensorflow as tf
from preprocess_utils import extract_mel_spectrogram


def load_audio_spectrogram(video_path, label):
    """Load and preprocess audio for a single video."""
    def _parse(video_path, label):
        video_path = video_path.numpy().decode('utf-8')
        spec = extract_mel_spectrogram(video_path)
        return spec.astype('float32'), label

    spec, label = tf.py_function(
        func=_parse,
        inp=[video_path, label],
        Tout=(tf.float32, tf.int32)
    )
    spec.set_shape((128, 128, 1))
    label.set_shape(())
    return spec, label


def _augment_spectrogram(spec, label):
    """SpecAugment-style augmentation for audio spectrograms.

    Implements:
    - Time shifting (circular roll)
    - Gaussian noise injection
    - Frequency masking (zero out random frequency bands)
    - Time masking (zero out random time segments)
    - Random gain adjustment
    """
    H = tf.shape(spec)[0]  # frequency bins
    W = tf.shape(spec)[1]  # time frames

    # 1) Time shift (circular roll ±10 frames)
    shift = tf.random.uniform([], -10, 11, dtype=tf.int32)
    spec = tf.roll(spec, shift=shift, axis=1)

    # 2) Gaussian noise
    noise = tf.random.normal(tf.shape(spec), stddev=3.0, dtype=spec.dtype)
    spec = spec + noise

    # 3) Frequency masking (SpecAugment F) — mask 1-2 bands
    num_freq_masks = tf.random.uniform([], 1, 3, dtype=tf.int32)
    for _ in range(2):  # max 2 masks
        f = tf.random.uniform([], 1, 16, dtype=tf.int32)  # mask width
        f0 = tf.random.uniform([], 0, H - f, dtype=tf.int32)
        # Create mask
        mask_indices = tf.range(f0, f0 + f)
        mask_indices = tf.clip_by_value(mask_indices, 0, H - 1)
        freq_mask = tf.ones([H, W, 1], dtype=spec.dtype)
        updates = tf.zeros([f, W, 1], dtype=spec.dtype)
        indices = tf.reshape(mask_indices, [-1, 1])
        freq_mask = tf.tensor_scatter_nd_update(freq_mask, indices,
                                                  tf.zeros([f, W, 1], dtype=spec.dtype))
        spec = spec * freq_mask + 127.5 * (1.0 - freq_mask)

    # 4) Time masking (SpecAugment T) — mask 1-2 segments
    for _ in range(2):
        t = tf.random.uniform([], 1, 12, dtype=tf.int32)  # mask width
        t0 = tf.random.uniform([], 0, W - t, dtype=tf.int32)
        time_mask = tf.ones([H, W, 1], dtype=spec.dtype)
        t_indices = tf.range(t0, t0 + t)
        t_indices = tf.clip_by_value(t_indices, 0, W - 1)
        # Transpose trick: mask time axis
        spec_t = tf.transpose(spec, [1, 0, 2])  # (W, H, 1)
        t_mask = tf.ones([W, H, 1], dtype=spec.dtype)
        t_mask = tf.tensor_scatter_nd_update(t_mask, tf.reshape(t_indices, [-1, 1]),
                                               tf.zeros([t, H, 1], dtype=spec.dtype))
        spec_t = spec_t * t_mask + 127.5 * (1.0 - t_mask)
        spec = tf.transpose(spec_t, [1, 0, 2])  # back to (H, W, 1)

    # 5) Random gain adjustment
    gain = tf.random.uniform([], 0.8, 1.2)
    mean_val = tf.reduce_mean(spec)
    spec = (spec - mean_val) * gain + mean_val

    spec = tf.clip_by_value(spec, 0.0, 255.0)
    return spec, label


def create_audio_dataset(
    manifest_path,
    batch_size=16,
    shuffle=True,
    augment=False,
    cache=False,
):
    """Creates a tf.data.Dataset for the audio expert."""
    df = pd.read_csv(manifest_path)
    file_paths = df['video_path'].values
    labels = df['label'].values.astype('int32')

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 10000))

    dataset = dataset.map(
        lambda p, l: load_audio_spectrogram(p, l),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        dataset = dataset.map(_augment_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache(cache) if isinstance(cache, str) else dataset.cache()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    ds = create_audio_dataset("data_manifest.csv", batch_size=4)
    for spec, labels in ds.take(1):
        print("Batch spectrogram shape:", spec.shape)
        print("Batch labels:", labels.numpy())
