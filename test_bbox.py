import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

rgb_frame = np.zeros((224, 224, 3), dtype=np.uint8)
rgb_frame[50:100, 50:100] = 255
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
res = detector.detect(mp_image)
if res.detections:
    det = res.detections[0]
    print(dir(det))
    print(dir(det.bounding_box))
    print(det.bounding_box.origin_x)
