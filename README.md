# Deepfake Image & Video Detector (FaceForensics++)

EfficientNet-B0 (ImageNet pre-trained, fine-tuned) classifies **face crops** as REAL / FAKE.
Videos are scored by sampling frames and averaging the per-frame fake probability.

## Pipeline

| Step | File               | What it does                                                                                                                  |
| ---- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1    | `extract_faces.py` | Videos/images -> face crops, labels from folder names, **identity-based** train/val/test split (no leakage)                   |
| 2    | `train.py`         | Fine-tunes the model, balanced sampling, augmentation (flip, colour, blur, JPEG), early stopping, saves `checkpoints/best.pt` |
| 3    | `evaluate.py`      | Test accuracy / precision / recall / F1 / AUC, frame- AND video-level, confusion matrix, ROC                                  |
| 4    | `detector.py`      | `DeepfakeDetector` class for your backend                                                                                     |
| 5    | `api_example.py`   | FastAPI example (`/predict/image`, `/predict/video`)                                                                          |

## Run on Kaggle (recommended - dataset is already there, free GPU)

1. New Notebook -> **Add Input** -> search `hungle3401/faceforensics` -> Settings: **Accelerator = GPU T4**.
2. Upload this folder as a Kaggle dataset (or paste the files), then in a cell:

```bash
!pip install facenet-pytorch --no-deps -q
!pip install timm -q
%cd /kaggle/working/deepfake_detector
!ls /kaggle/input/faceforensics            # check the real folder name/structure
!python extract_faces.py --raw_dir /kaggle/input/faceforensics --max_per_label 150   # quick test first
!python train.py --epochs 2                                                          # quick test
# Happy? Re-run with everything:
!python extract_faces.py --raw_dir /kaggle/input/faceforensics
!python train.py --epochs 10
!python evaluate.py
```

3. Download `checkpoints/best.pt` (+ the png/json results for your report).

Colab / local GPU works the same: download the dataset, set `--raw_dir` to its path.

## Use in your backend

```python
from detector import DeepfakeDetector
det = DeepfakeDetector("checkpoints/best.pt")          # once at startup
det.predict_image("a.jpg")   # or bytes
det.predict_video("b.mp4", num_frames=16)
```

Example response:

```json
{"type":"video","label":"FAKE","confidence":0.93,"fake_probability":0.93,"threshold":0.48,
 "face_found":true,"frames_analyzed":16,"frame_probabilities":[0.91,0.95,...]}
```

`frame_probabilities` is handy for a timeline chart in the frontend.
Backend needs: `best.pt`, `detector.py`, `face_utils.py`, `dataset.py`, `model.py`.

## Notes for your report / viva

- **Dataset caveat:** labels are inferred from folder names - run `ls` on the dataset and make sure `original*` = real, manipulation folders = fake. `extract_faces.py` prints any folders it skipped.
- **Class imbalance** (few real, many fake videos) handled by more frames per real video + balanced sampler.
- **Generalisation limit:** a model trained on FF++ works best on similar face-swap/reenactment fakes; it can be weaker on fully AI-generated images (GAN/diffusion) or very different compression. State this as a limitation / future work.
- Metrics to report: accuracy, F1, ROC-AUC (frame- and video-level), confusion matrix.
- Possible extensions: Grad-CAM heatmaps, temporal model (LSTM) over frame features, XceptionNet comparison (`--model xception`).
