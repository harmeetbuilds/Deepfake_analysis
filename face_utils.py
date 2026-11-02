"""Face detection + cropping. MTCNN (facenet-pytorch) with an OpenCV Haar fallback."""
import cv2
import numpy as np
import torch
from PIL import Image


class FaceDetector:
    def __init__(self, device=None, margin=0.3, size=224, min_prob=0.9, detect_max_side=640):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.margin, self.size, self.min_prob = margin, size, min_prob
        self.detect_max_side = detect_max_side
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(keep_all=True, device=self.device)
            self.backend = "mtcnn"
        except Exception:
            self.mtcnn = None
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.backend = "haar"
            print("[FaceDetector] facenet-pytorch not found -> using weaker Haar fallback")

    def _boxes(self, rgb):
        """Return list of (x1,y1,x2,y2) in original-image coordinates."""
        h, w = rgb.shape[:2]
        scale = min(1.0, self.detect_max_side / max(h, w))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1 else rgb
        if self.backend == "mtcnn":
            boxes, probs = self.mtcnn.detect(Image.fromarray(small))
            if boxes is None:
                return []
            out = [b for b, p in zip(boxes, probs) if p is not None and p >= self.min_prob]
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            f = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            out = [(x, y, x + bw, y + bh) for x, y, bw, bh in f]
        return [tuple(float(v) / scale for v in b) for b in out]

    def crop(self, rgb):
        """Crop the largest face (with margin) -> RGB uint8 array (size x size), or None."""
        boxes = self._boxes(rgb)
        if not boxes:
            return None
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        bw, bh = x2 - x1, y2 - y1
        h, w = rgb.shape[:2]
        x1, x2 = max(0, int(x1 - bw * self.margin)), min(w, int(x2 + bw * self.margin))
        y1, y2 = max(0, int(y1 - bh * self.margin)), min(h, int(y2 + bh * self.margin))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        return cv2.resize(rgb[y1:y2, x1:x2], (self.size, self.size), interpolation=cv2.INTER_AREA)
