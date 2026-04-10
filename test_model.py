# test_model.py
import tensorflow as tf
import pandas as pd
from preprocess_utils import extract_and_crop_faces

# Load trained model
model = tf.keras.models.load_model("visual_expert_best.h5")

# Pick a video from validation set
df = pd.read_csv("val_manifest.csv")
sample = df.sample(5).iloc[1]
video_path = sample['video_path']
true_label = sample['label']

# Preprocess
frames = extract_and_crop_faces("/Users/kanwal/Desktop/Deep-Fake2.0/933.mp4", num_frames=10)
frames = tf.expand_dims(frames, axis=0)  # Add batch dimension

# Predict
pred = model.predict(frames, verbose=0)[0][0]
pred_label = 1 if pred > 0.5 else 0
confidence = pred if pred > 0.5 else 1 - pred

print(pred_label)
'''
print(f"Video: {video_path}")
print(f"True label: {'REAL' if true_label == 1 else 'FAKE'}")
print(f"Predicted: {'REAL' if pred_label == 1 else 'FAKE'} (confidence: {confidence:.2%})")'''