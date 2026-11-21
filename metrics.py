import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    probs, labels, paths = [], [], []
    for x, y, p in loader:
        probs += torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu().tolist()
        labels += y.tolist()
        paths += list(p)
    return np.array(probs), np.array(labels), paths


def best_threshold(labels, probs):
    """Youden's J on the ROC curve."""
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[np.argmax(tpr - fpr)])


def compute_metrics(labels, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    return dict(
        accuracy=float(accuracy_score(labels, pred)),
        precision=float(precision_score(labels, pred, zero_division=0)),
        recall=float(recall_score(labels, pred, zero_division=0)),
        f1=float(f1_score(labels, pred, zero_division=0)),
        auc=float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        threshold=float(thr),
    )
