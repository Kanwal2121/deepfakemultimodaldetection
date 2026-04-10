import os
import pandas as pd
import numpy as np
from lip_sync_analyzer import LipSyncAnalyzer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_video(video_info):
    """Worker function to process a single video in its own process."""
    video_path, label = video_info
    try:
        # Re-initialize analyzer inside the process for thread safety
        analyzer = LipSyncAnalyzer()
        score = analyzer.compute_sync_score(video_path)
        return score, label
    except Exception:
        return None, label

def find_best_threshold(real_scores, fake_scores):
    """Search for the threshold that maximizes classification accuracy."""
    thresholds = np.linspace(0, 1, 101)
    best_acc = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        real_correct = sum(s >= thresh for s in real_scores)
        fake_correct = sum(s < thresh for s in fake_scores)
        acc = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc

if __name__ == "__main__":
    # 1. Setup Data
    if not os.path.exists("data_manifest.csv"):
        print("❌ Error: data_manifest.csv not found.")
        exit(1)
        
    df = pd.read_csv("data_manifest.csv")
    
    # 2. Stratified Sampling (100 Real, 100 Fake)
    real_subset = df[df['label'] == 1]
    fake_subset = df[df['label'] == 0]
    
    n_samples = min(100, len(real_subset), len(fake_subset))
    df_real = real_subset.sample(n=n_samples, random_state=42)
    df_fake = fake_subset.sample(n=n_samples, random_state=42)
    df_sample = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42)
    
    video_tasks = [(row['video_path'], row['label']) for _, row in df_sample.iterrows()]
    
    # 3. Ensure model is downloaded before spawning processes
    print("⏳ Initializing environment...")
    LipSyncAnalyzer() 
    
    # 4. Parallel Evaluation
    real_scores = []
    fake_scores = []
    failures = 0
    
    print(f"🚀 Starting parallel evaluation of {len(video_tasks)} videos using 4 workers...")
    print("   (This will take a few minutes. Check the progress bar below.)")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_video, task): task for task in video_tasks}
        
        for future in tqdm(as_completed(futures), total=len(video_tasks), desc="Processing"):
            score, label = future.result()
            if score is not None:
                if label == 1:
                    real_scores.append(score)
                else:
                    fake_scores.append(score)
            else:
                failures += 1

    # 5. Output Results
    print("\n" + "="*45)
    print("📊 LIP-SYNC EVALUATION RESULTS")
    print("="*45)
    
    if real_scores:
        print(f"✅ Real Videos (n={len(real_scores)}): mean={np.mean(real_scores):.3f}, std={np.std(real_scores):.3f}")
    if fake_scores:
        print(f"❌ Fake Videos (n={len(fake_scores)}): mean={np.mean(fake_scores):.3f}, std={np.std(fake_scores):.3f}")
    print(f"⚠️ Failures (Skipped): {failures}")
    
    if real_scores and fake_scores:
        best_thresh, best_acc = find_best_threshold(real_scores, fake_scores)
        print("-" * 45)
        print(f"🎯 OPTIMAL THRESHOLD: {best_thresh:.3f}")
        print(f"📈 MAX ACCURACY:     {best_acc:.1%}")
        print("-" * 45)
        print(f"💡 Tip: Scores >= {best_thresh:.3f} are likely REAL.")
    else:
        print("\n❌ Evaluation failed: Not enough successful data points.")
    print("="*45)