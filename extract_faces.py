"""
Step 1: Turn the raw FaceForensics dataset (videos OR images) into a face-crop dataset:

  data/processed/{train,val,test}/{real,fake}/*.jpg

Labels are inferred from folder names (original/real -> real; Deepfakes, Face2Face,
FaceSwap, NeuralTextures, FaceShifter, fake... -> fake).
Splitting is done by *source person ID* so the same face never appears in both train and test.

Usage:
  python extract_faces.py --raw_dir /kaggle/input/faceforensics --out_dir data/processed
  python extract_faces.py --raw_dir ... --max_per_label 200      # quick smoke test
"""
import argparse, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from face_utils import FaceDetector

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
REAL_KEYS = ("original", "real", "pristine")
FAKE_KEYS = ("fake", "deepfake", "face2face", "faceswap", "neuraltextures",
             "faceshifter", "manipulated", "synthetic")


def infer_label(path: Path, root: Path):
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    if any(k in p for p in parts for k in REAL_KEYS):
        return 0
    if any(k in p for p in parts for k in FAKE_KEYS):
        return 1
    return None


def group_key(path: Path):
    """FF++: '000_003.mp4' (fake) and '000.mp4' (real) -> group '000'."""
    stem = re.sub(r"([_-]?(frame|f)?[_-]?\d+)$", "", path.stem) if path.suffix.lower() in IMG_EXT else path.stem
    return (stem or path.stem).split("_")[0]


def scan(root: Path):
    samples, skipped = [], Counter()
    for p in root.rglob("*"):
        ext = p.suffix.lower()
        if ext not in VIDEO_EXT | IMG_EXT:
            continue
        label = infer_label(p, root)
        if label is None:
            skipped[str(p.parent.relative_to(root))] += 1
            continue
        samples.append(dict(path=p, label=label, is_video=ext in VIDEO_EXT,
                            group=group_key(p),
                            key=str(p.relative_to(root).with_suffix("")).replace("/", "-").replace("\\", "-")))
    return samples, skipped


def assign_splits(samples, ratios, seed):
    groups = sorted({s["group"] for s in samples})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
    split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, g in enumerate(groups)}
    for s in samples:
        s["split"] = split_of[s["group"]]
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frames_real", type=int, default=20, help="frames per REAL video")
    ap.add_argument("--frames_fake", type=int, default=4, help="frames per FAKE video (more fake videos exist)")
    ap.add_argument("--max_per_label", type=int, default=0, help="limit #videos/images per label (0 = all)")
    ap.add_argument("--no_detect", action="store_true", help="dataset images are already face crops")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root, out = Path(a.raw_dir), Path(a.out_dir)
    samples, skipped = scan(root)
    if not samples:
        raise SystemExit("No labelled videos/images found. Check --raw_dir and folder names.")
    print("Folders skipped (label unknown):", dict(skipped) or "none")

    rng = random.Random(a.seed)
    if a.max_per_label:
        by = defaultdict(list)
        for s in samples:
            by[s["label"]].append(s)
        samples = [s for v in by.values() for s in rng.sample(v, min(len(v), a.max_per_label))]

    n_groups = assign_splits(samples, (0.70, 0.15, 0.15), a.seed)
    print(f"{len(samples)} items | {n_groups} identity groups | "
          f"{Counter((s['split'], s['label']) for s in samples)}")

    det = None if a.no_detect else FaceDetector(size=a.img_size)
    stats = Counter()

    def save(face_rgb, s, idx):
        d = out / s["split"] / ("fake" if s["label"] else "real")
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{s['key']}__f{idx:03d}.jpg"), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        stats[(s["split"], s["label"])] += 1

    for s in tqdm(samples, desc="Extracting"):
        if s["is_video"]:
            cap = cv2.VideoCapture(str(s["path"]))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                continue
            k = a.frames_fake if s["label"] else a.frames_real
            for i, fi in enumerate(np.unique(np.linspace(0, total - 1, k).astype(int))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = rgb_resize(rgb, a.img_size) if det is None else det.crop(rgb)
                if face is not None:
                    save(face, s, i)
            cap.release()
        else:
            bgr = cv2.imread(str(s["path"]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            face = rgb_resize(rgb, a.img_size) if det is None else det.crop(rgb)
            if face is not None:
                save(face, s, 0)

    summary = {f"{sp}/{'fake' if l else 'real'}": c for (sp, l), c in sorted(stats.items())}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved face crops:", json.dumps(summary, indent=2))


def rgb_resize(rgb, size):
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
