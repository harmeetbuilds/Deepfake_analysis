"""Settings - override any of them with environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", str(ROOT / "checkpoints" / "best.pt"))
    max_image_mb: int = int(os.getenv("MAX_IMAGE_MB", "10"))
    max_video_mb: int = int(os.getenv("MAX_VIDEO_MB", "100"))
    video_frames: int = int(os.getenv("VIDEO_FRAMES", "16"))
    # score within +-margin of the threshold is flagged "uncertain"
    uncertain_margin: float = float(os.getenv("UNCERTAIN_MARGIN", "0.15"))
    # e.g. ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
    allowed_origins: list = field(default_factory=lambda: _csv(os.getenv("ALLOWED_ORIGINS", "*")))


settings = Settings()
