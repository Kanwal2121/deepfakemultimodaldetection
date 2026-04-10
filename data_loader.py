# data_loader.py — Visual Data Pipeline (Production)
# Clean and fast — no complex tensor ops that slow down tf.data

import pandas as pd
import tensorflow as tf
from preprocess_utils import extract_and_crop_faces


def load_video_frames(video_path, label, num_frames=10):
    """Load and preprocess a single video for the visual model."""
    def _parse(video_path, label):
        video_path = video_path.numpy().decode('utf-8')
        frames = extract_and_crop_faces(video_path, num_frames=num_frames)
        return frames.astype('float32'), label

    frames, label = tf.py_function(
        func=_parse,
        inp=[video_path, label],
        Tout=(tf.float32, tf.int32)
    )
    frames.set_shape((num_frames, 224, 224, 3))
    label.set_shape(())
    return frames, label


def _augment_visual_frames(frames, label):
    """Effective but fast augmentation for deepfake face crops."""
    # Spatial
    frames = tf.image.random_flip_left_right(frames)
    frames = tf.image.random_brightness(frames, max_delta=15.0)
    frames = tf.image.random_contrast(frames, lower=0.80, upper=1.20)
    frames = tf.image.random_saturation(frames, lower=0.85, upper=1.15)

    # Gaussian noise (simulates compression / sensor noise)
    noise = tf.random.normal(tf.shape(frames), stddev=4.0, dtype=frames.dtype)
    frames = tf.clip_by_value(frames + noise, 0.0, 255.0)

    return frames, label


def create_visual_dataset(
    manifest_path,
    batch_size=8,
    num_frames=10,
    shuffle=True,
    augment=False,
    cache=False,
):
    """Creates a tf.data.Dataset for the visual expert."""
    df = pd.read_csv(manifest_path)
    file_paths = df['video_path'].values
    labels = df['label'].values.astype('int32')

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 10000))

    dataset = dataset.map(
        lambda p, l: load_video_frames(p, l, num_frames=num_frames),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        dataset = dataset.map(_augment_visual_frames, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache(cache) if isinstance(cache, str) else dataset.cache()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    ds = create_visual_dataset("data_manifest.csv", batch_size=2)
    for frames, labels in ds.take(1):
        print("Batch frames shape:", frames.shape)
        print("Batch labels:", labels.numpy())
