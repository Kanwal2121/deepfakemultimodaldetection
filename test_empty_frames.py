import pandas as pd
from preprocess_utils import extract_and_crop_faces
import numpy as np

df = pd.read_csv("train_manifest.csv")
for i in range(2):
    path = df['video_path'].iloc[i]
    frames = extract_and_crop_faces(path)
    print(f"Video {i}: max pixel = {np.max(frames)}, min pixel = {np.min(frames)}, mean = {np.mean(frames)}")
