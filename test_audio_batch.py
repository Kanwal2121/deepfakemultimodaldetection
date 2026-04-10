# test_audio_batch.py
import tensorflow as tf
import pandas as pd
import numpy as np
from preprocess_utils import extract_mel_spectrogram

model = tf.keras.models.load_model("audio_expert_best.h5")
df = pd.read_csv("val_audio_manifest.csv")

# Test on 10 random videos
samples = df.sample(10)
correct = 0

for _, row in samples.iterrows():
    video_path = row['video_path']
    true_label = row['label']
    
    spec = extract_mel_spectrogram(video_path)
    spec = tf.expand_dims(spec, axis=0)
    
    pred = model.predict(spec, verbose=0)[0][0]
    pred_label = 1 if pred > 0.5 else 0
    
    if pred_label == true_label:
        correct += 1
        status = "✅"
    else:
        status = "❌"
    
    print(f"{status} True: {true_label} | Pred: {pred_label} | Conf: {pred:.3f}")

print(f"\nAccuracy on 10 samples: {correct/10:.1%}")
# check_balance.py
import pandas as pd
val_df = pd.read_csv("val_audio_manifest.csv")
print(val_df['label'].value_counts())