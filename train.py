"""
Step 2: Train.   python train.py --data_dir data/processed --epochs 10
Saves best model (by validation AUC) to checkpoints/best.pt
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import FaceDataset, get_eval_transform, get_train_transform
from metrics import best_threshold, collect_predictions, compute_metrics
from model import build_model


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    tr = FaceDataset(Path(a.data_dir) / "train", get_train_transform(a.img_size))
    va = FaceDataset(Path(a.data_dir) / "val", get_eval_transform(a.img_size))
    print(f"train={len(tr)} val={len(va)} device={dev}")

    # Balance real/fake in every batch (there are far more fake videos than real)
    counts = np.bincount(tr.labels, minlength=2)
    w = [1.0 / counts[l] for l in tr.labels]
    sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
    tl = DataLoader(tr, a.batch_size, sampler=sampler, num_workers=a.workers, pin_memory=amp, drop_last=True)
    vl = DataLoader(va, a.batch_size * 2, shuffle=False, num_workers=a.workers, pin_memory=amp)

    model = build_model(a.model).to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_auc, bad, history = -1, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for x, y, _ in tqdm(tl, desc=f"Epoch {ep}/{a.epochs}", leave=False):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item() * len(x); n += len(x)

        probs, labels, _ = collect_predictions(model, vl, dev)
        m = compute_metrics(labels, probs, 0.5)
        m.update(epoch=ep, train_loss=run / n)
        history.append(m)
        print(f"ep{ep}: loss={m['train_loss']:.4f} val_acc={m['accuracy']:.4f} "
              f"val_auc={m['auc']:.4f} val_f1={m['f1']:.4f}")

        if m["auc"] > best_auc:
            best_auc, bad = m["auc"], 0
            thr = best_threshold(labels, probs)
            torch.save(dict(model_state=model.state_dict(), model_name=a.model,
                            img_size=a.img_size, threshold=thr, val_metrics=m), out / "best.pt")
            print(f"  saved best.pt (val AUC {best_auc:.4f}, threshold {thr:.3f})")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping."); break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(e, [h["train_loss"] for h in history]); ax[0].set_title("Train loss")
        ax[1].plot(e, [h["accuracy"] for h in history], label="val acc")
        ax[1].plot(e, [h["auc"] for h in history], label="val AUC"); ax[1].legend(); ax[1].set_title("Validation")
        for x_ in ax: x_.set_xlabel("epoch")
        plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=150)
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
