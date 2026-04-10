# test_lip_sync.py
import pandas as pd
from lip_sync_analyzer import LipSyncAnalyzer

analyzer = LipSyncAnalyzer()
df = pd.read_csv("data_manifest.csv")

# Test on one real and one fake
real_video = df[df['label'] == 1].iloc[0]['video_path']
fake_video = df[df['label'] == 0].iloc[0]['video_path']

score_real = analyzer.compute_sync_score(real_video)
score_fake = analyzer.compute_sync_score(fake_video)

print(f"Real video sync score: {score_real:.3f}")
print(f"Fake video sync score: {score_fake:.3f}")
print("\n💡 Expect real video to have HIGHER sync score (>0.6)")