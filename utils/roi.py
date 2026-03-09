import time

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


class RegionExtractor:
    def __init__(self, mode="face", fallback_mode="smart_crop"):
        self.mode = mode
        self.fallback_mode = fallback_mode
        self.face_cascade = None
        if cv2 is not None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.face_cascade = cascade

    def extract(self, image):
        start_time = time.perf_counter()
        image = image.convert("RGB")
        bbox = None
        detector_used = "none"

        if self.mode == "face" and self.face_cascade is not None:
            bbox = self._detect_face_bbox(image)
            detector_used = "haar_face"

        if bbox is None:
            bbox = self._fallback_bbox(image)
            detector_used = self.fallback_mode if bbox is not None else detector_used

        if bbox is None:
            roi_image = image
            roi_applied = False
            bbox_data = None
        else:
            roi_image = image.crop(bbox)
            roi_applied = bbox != (0, 0, image.width, image.height)
            bbox_data = {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]}

        return roi_image, {
            "mode": self.mode,
            "fallback_mode": self.fallback_mode,
            "detector_used": detector_used,
            "face_detector_available": self.face_cascade is not None,
            "roi_applied": roi_applied,
            "bbox": bbox_data,
            "source_size": {"width": image.width, "height": image.height},
            "roi_size": {"width": roi_image.width, "height": roi_image.height},
            "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

    def _detect_face_bbox(self, image):
        image_array = np.array(image)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return self._expand_bbox(image.width, image.height, x, y, w, h)

    def _fallback_bbox(self, image):
        width, height = image.size
        if self.fallback_mode == "smart_crop":
            crop_size = int(min(width, height) * 0.82)
            left = max(0, (width - crop_size) // 2)
            top = max(0, int((height - crop_size) * 0.28))
            top = min(top, max(0, height - crop_size))
            return (left, top, left + crop_size, top + crop_size)
        if self.fallback_mode == "center_crop":
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            return (left, top, left + crop_size, top + crop_size)
        if self.fallback_mode == "full_image":
            return None
        return None

    @staticmethod
    def _expand_bbox(image_width, image_height, x, y, w, h):
        center_x = x + w / 2
        center_y = y + h / 2
        size = int(max(w, h) * 1.9)
        left = max(0, int(center_x - size / 2))
        top = max(0, int(center_y - size * 0.42))
        right = min(image_width, left + size)
        bottom = min(image_height, top + size)

        if right - left < size:
            left = max(0, right - size)
        if bottom - top < size:
            top = max(0, bottom - size)
        return (left, top, right, bottom)
