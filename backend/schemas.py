from typing import Dict, List, Literal, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str
    type: Literal["image", "video"]
    label: Literal["REAL", "FAKE"]
    confidence: float            # confidence in the predicted label (0-1)
    fake_probability: float      # raw model score (0 = real, 1 = fake)
    threshold: float
    uncertain: bool              # True when the score is close to the threshold
    face_found: bool
    frames_analyzed: int
    frame_probabilities: List[float]   # per-frame scores (a single value for images)
    inference_ms: int
    warning: Optional[str] = None


class ModelInfo(BaseModel):
    model_name: str
    img_size: int
    threshold: float
    device: str
    face_detector: str
    val_metrics: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
