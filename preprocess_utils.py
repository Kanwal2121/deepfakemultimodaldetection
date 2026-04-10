
import os
import ssl
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import librosa
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
if not os.path.exists(model_path):
    context = ssl._create_unverified_context()
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", model_path, context=context)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
face_detector = vision.FaceDetector.create_from_options(options)

def extract_and_crop_faces(video_path, num_frames=10, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return np.zeros((num_frames, *target_size, 3), dtype=np.float32)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    cropped_faces = []
    last_valid_face = None

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_detector.detect(mp_image)

        if results.detections:
            largest_detection = max(results.detections, key=lambda d: d.categories[0].score)
            bbox = largest_detection.bounding_box
            h, w, _ = frame.shape
            x = bbox.origin_x
            y = bbox.origin_y
            bw = bbox.width
            bh = bbox.height

            margin = 0.2
            x = max(0, int(x - bw * margin))
            y = max(0, int(y - bh * margin))
            x2 = min(w, int(x + bw * (1 + 2 * margin)))
            y2 = min(h, int(y + bh * (1 + 2 * margin)))

            cropped_face = rgb_frame[y:y2, x:x2]
            if cropped_face.size > 0:
                cropped_face = cv2.resize(cropped_face, target_size).astype(np.float32)
                cropped_faces.append(cropped_face)
                last_valid_face = cropped_face
                continue

        if last_valid_face is not None:
            cropped_faces.append(last_valid_face.copy())
        else:
            cropped_faces.append(np.zeros((*target_size, 3), dtype=np.float32))

    cap.release()

    while len(cropped_faces) < num_frames:
        if last_valid_face is not None:
            cropped_faces.append(last_valid_face.copy())
        else:
            cropped_faces.append(np.zeros((*target_size, 3), dtype=np.float32))

    return np.array(cropped_faces, dtype=np.float32)

def extract_mel_spectrogram(video_path, target_shape=(128, 128), duration_sec=4.0):
    try:
        y, sr = librosa.load(video_path, sr=16000, mono=True)
        if len(y) == 0:
            return np.full((*target_shape, 1), 0.5, dtype=np.float32)

        target_len = int(sr * duration_sec)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        elif len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=target_shape[0],
            n_fft=1024,
            hop_length=160,
            win_length=400,
            fmin=20,
            fmax=8000,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_min = np.min(mel_spec_db)
        mel_max = np.max(mel_spec_db)
        mel_spec_norm = (mel_spec_db - mel_min) / (mel_max - mel_min + 1e-8)
        mel_spec_norm = np.clip(mel_spec_norm, 0.0, 1.0) * 255.0

        mel_spec_resized = cv2.resize(mel_spec_norm, target_shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(mel_spec_resized, axis=-1).astype(np.float32)

    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return np.full((*target_shape, 1), 0.5, dtype=np.float32)
