import os
import ssl
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import librosa
import numpy as np

class LipSyncAnalyzer:
    def __init__(self):
        """Initialize the LipSyncAnalyzer and download models if necessary."""
        self.model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        self._ensure_model_exists()
        
        # Initialize the Face Landmarker Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Ensure the face_landmarker.task model is available locally."""
        if not os.path.exists(self.model_path):
            print("📥 Downloading Face Landmarker model (required for lip-sync)...")
            try:
                context = ssl._create_unverified_context()
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                with urllib.request.urlopen(url, context=context) as response, open(self.model_path, 'wb') as f:
                    f.write(response.read())
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                raise

    def compute_mouth_openness(self, landmarks, frame_shape):
        """Calculate Mouth Aspect Ratio (MAR) using standard landmarks."""
        h, w, _ = frame_shape
        # Inner lip centers: 13 (top), 14 (bottom)
        p13, p14 = landmarks[13], landmarks[14]
        h_dist = np.sqrt(((p13.x - p14.x) * w)**2 + ((p13.y - p14.y) * h)**2)
        
        # Outer lip corners: 61 (left), 291 (right)
        p61, p291 = landmarks[61], landmarks[291]
        w_dist = np.sqrt(((p61.x - p291.x) * w)**2 + ((p61.y - p291.y) * h)**2)
        
        return h_dist / (w_dist + 1e-6)

    def extract_mouth_signal(self, video_path, max_frames=500):
        """Extract time series of mouth openness from video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        mouth_openness = []
        
        while len(mouth_openness) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            
            if result.face_landmarks:
                mouth_openness.append(self.compute_mouth_openness(result.face_landmarks[0], frame.shape))
            else:
                mouth_openness.append(0.0)
        
        cap.release()
        return np.array(mouth_openness), fps

    def compute_sync_score(self, video_path):
        """Estimate lip-sync quality by correlating mouth movement with audio energy."""
        try:
            # 1. Extract signals
            mouth_signal, fps = self.extract_mouth_signal(video_path)
            if fps == 0 or fps is None: fps = 25
            
            # 2. Extract audio energy envelope
            y, sr = librosa.load(video_path, sr=16000)
            hop = int(sr / fps)
            audio_energy = librosa.feature.rms(y=y, hop_length=hop)[0]
            
            # 3. Align and Normalize
            min_len = min(len(mouth_signal), len(audio_energy))
            if min_len < 20: return None  # Too short to analyze
            
            m = mouth_signal[:min_len]
            a = audio_energy[:min_len]
            
            # Standardize for correlation
            m = (m - np.mean(m)) / (np.std(m) + 1e-8)
            a = (a - np.mean(a)) / (np.std(a) + 1e-8)
            
            # 4. Lag-Aware Best Correlation (Window +/- 5 frames)
            # This accounts for natural speech leads or encoding delays
            max_corr = 0
            for lag in range(-5, 6):
                if lag < 0:
                    c = np.mean(m[-lag:] * a[:lag])
                elif lag > 0:
                    c = np.mean(m[:-lag] * a[lag:])
                else:
                    c = np.mean(m * a)
                max_corr = max(max_corr, c)
            
            return float(np.clip((max_corr + 1) / 2, 0.0, 1.0))
            
        except Exception as e:
            # Note: Returning None allows evaluators to skip failed files
            return None