"""
Inference module - THIS is what your backend imports.

    from detector import DeepfakeDetector
    det = DeepfakeDetector("checkpoints/best.pt")      # load ONCE at server startup
    det.predict_image("photo.jpg")  or  det.predict_image(file_bytes)
    det.predict_video("clip.mp4", num_frames=16)

Both return a JSON-serialisable dict:
  {"type", "label": "FAKE"|"REAL", "confidence", "fake_probability", "threshold",
   "face_found", "frames_analyzed", "frame_probabilities": [...]}
"""
import cv2
import numpy as np
import torch
from PIL import Image

from dataset import get_eval_transform
from face_utils import FaceDetector
from model import build_model


class DeepfakeDetector:
    def __init__(self, ckpt_path="checkpoints/best.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device)
        self.model = build_model(ck["model_name"], pretrained=False).to(self.device).eval()
        self.model.load_state_dict(ck["model_state"])
        self.threshold = float(ck.get("threshold", 0.5))
        self.size = ck["img_size"]
        self.tf = get_eval_transform(self.size)
        self.faces = FaceDetector(device=self.device, size=self.size)
        self.info = dict(model_name=ck["model_name"], img_size=self.size, threshold=self.threshold,
                         device=self.device, face_detector=self.faces.backend,
                         val_metrics={k: round(v, 4) for k, v in ck.get("val_metrics", {}).items()
                                      if isinstance(v, float)})

    # ---------- helpers ----------
    @torch.no_grad()
    def _score(self, crops):
        x = torch.stack([self.tf(Image.fromarray(c)) for c in crops]).to(self.device)
        out = []
        for i in range(0, len(x), 32):
            out += torch.sigmoid(self.model(x[i:i + 32]).squeeze(1)).float().cpu().tolist()
        return out

    def _result(self, kind, probs, face_found):
        p = float(np.mean(probs))
        fake = p >= self.threshold
        return dict(type=kind, label="FAKE" if fake else "REAL",
                    confidence=round(p if fake else 1 - p, 4), fake_probability=round(p, 4),
                    threshold=round(self.threshold, 4), face_found=face_found,
                    frames_analyzed=len(probs), frame_probabilities=[round(float(q), 4) for q in probs])

    # ---------- public API ----------
    def predict_image(self, source):
        """source: file path (str/Path), raw bytes, PIL.Image or RGB numpy array."""
        if isinstance(source, (bytes, bytearray)):
            bgr = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not read image (corrupt or unsupported format)")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(source, Image.Image):
            rgb = np.array(source.convert("RGB"))
        elif isinstance(source, np.ndarray):
            rgb = source
        else:
            bgr = cv2.imread(str(source))
            if bgr is None:
                raise ValueError("Could not read image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = self.faces.crop(rgb)
        found = crop is not None
        if not found:  # fall back to whole image (result is less reliable)
            crop = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return self._result("image", self._score([crop]), found)

    def predict_video(self, path, num_frames=16):
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("Could not read video")
        crops, fallback = [], []
        for fi in np.unique(np.linspace(0, total - 1, num_frames).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            c = self.faces.crop(rgb)
            if c is not None:
                crops.append(c)
            else:
                fallback.append(cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA))
        cap.release()
        found = len(crops) > 0
        use = crops if found else fallback
        if not use:
            raise ValueError("No frames could be decoded")
        return self._result("video", self._score(use), found)
