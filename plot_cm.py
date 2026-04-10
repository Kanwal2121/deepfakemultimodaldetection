import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from data_loader import create_visual_dataset

print("Loading model...")
if os.path.exists("visual_expert_best.keras"):
    model = tf.keras.models.load_model("visual_expert_best.keras")
else:
    model = tf.keras.models.load_model("visual_expert_best.h5")

print("Loading validation dataset...")
# Make sure shuffle=False to match labels
val_ds = create_visual_dataset("val_manifest.csv", batch_size=8, shuffle=False)

print("Predicting...")
preds = model.predict(val_ds)
y_pred = (preds > 0.5).astype(int).flatten()

print("Fetching true labels...")
df = pd.read_csv("val_manifest.csv")
if "video_label" in df.columns:
    df["label"] = df["video_label"].astype(int)
y_true = df['label'].values

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Fake (0)', 'Real (1)']))

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Visual Expert')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Fake', 'Real'])
plt.yticks(tick_marks, ['Fake', 'Real'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Saved plot to /Users/kanwal/Desktop/Deep-Fake2.0/confusion_matrix.png")
