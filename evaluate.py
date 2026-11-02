"""
Step 3: Evaluate best.pt on the held-out TEST split (frame-level and video-level).
  python evaluate.py --data_dir data/processed --ckpt checkpoints/best.pt
Outputs: checkpoints/test_metrics.json, confusion_matrix.png, roc_curve.png
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from dataset import FaceDataset, get_eval_transform
from metrics import collect_predictions, compute_metrics
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(ck["model_name"], pretrained=False).to(dev)
    model.load_state_dict(ck["model_state"])
    thr = ck["threshold"]

    ds = FaceDataset(Path(a.data_dir) / "test", get_eval_transform(ck["img_size"]))
    dl = DataLoader(ds, a.batch_size, shuffle=False, num_workers=a.workers)
    probs, labels, paths = collect_predictions(model, dl, dev)

    res = {"frame_level@0.5": compute_metrics(labels, probs, 0.5),
           "frame_level@tuned": compute_metrics(labels, probs, thr)}

    # Video-level: average frame probabilities of each source video
    vid = defaultdict(list); vlab = {}
    for p, y, path in zip(probs, labels, paths):
        k = Path(path).name.rsplit("__f", 1)[0]
        vid[k].append(p); vlab[k] = y
    vp = np.array([np.mean(v) for v in vid.values()]); vy = np.array([vlab[k] for k in vid])
    if len(set(vy)) > 1:
        res["video_level@0.5"] = compute_metrics(vy, vp, 0.5)
        res["n_videos"] = len(vy)

    out = Path(a.ckpt).parent
    (out / "test_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(labels, (probs >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["real", "fake"]).plot(cmap="Blues")
    plt.title("Confusion matrix (test, frame-level)"); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr, label=f"AUC={res['frame_level@0.5']['auc']:.3f}"); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC (test)")
    plt.savefig(out / "roc_curve.png", dpi=150)


if __name__ == "__main__":
    main()
