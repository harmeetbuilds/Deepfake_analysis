"""
Test the trained model on whole VIDEOS.

1) Held-out test videos from the dataset (never seen in training):
     python test_videos.py --n 10
   -> picks 10 real + 10 fake videos from the TEST split and checks the predictions.

2) Your own file or folder of videos/images:
     python test_videos.py --path my_clip.mp4
     python test_videos.py --path my_videos_folder --expect real
   (--expect real|fake is optional; it just lets the script count right/wrong.)

Results are also saved to checkpoints/video_test_results.csv
"""
import argparse, csv, random, time
from pathlib import Path

from detector import DeepfakeDetector

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def test_split_videos(processed, raw):
    """Map the crops in data/processed/test back to the original video files."""
    vids = {"real": [], "fake": []}
    for label in vids:
        keys = {f.name.rsplit("__f", 1)[0] for f in (Path(processed) / "test" / label).glob("*.jpg")}
        for k in sorted(keys):
            folder, stem = k.split("-", 1)          # key = "<folder>-<video name>"
            p = Path(raw) / folder / f"{stem}.mp4"
            if not p.exists():
                p = next((Path(raw) / folder).glob(stem + ".*"), None)
            if p and Path(p).exists():
                vids[label].append(Path(p))
    return vids


def run(det, path, frames):
    t = time.time()
    r = det.predict_image(path) if path.suffix.lower() in IMG_EXT else det.predict_video(path, frames)
    r["seconds"] = round(time.time() - t, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--raw", default="FF++")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--n", type=int, default=10, help="videos per class from the test split")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--path", help="your own video/image file or a folder")
    ap.add_argument("--expect", choices=["real", "fake"], help="true label of --path files (optional)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    det = DeepfakeDetector(a.ckpt)
    jobs = []  # (path, true_label or None)
    if a.path:
        p = Path(a.path)
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXT | IMG_EXT)
        jobs = [(f, a.expect) for f in files]
    else:
        vids = test_split_videos(a.processed, a.raw)
        print({k: len(v) for k, v in vids.items()}, "test videos available")
        rng = random.Random(a.seed)
        for label, lst in vids.items():
            jobs += [(v, label) for v in rng.sample(lst, min(a.n, len(lst)))]
    if not jobs:
        raise SystemExit("Nothing to test.")

    rows, tp = [], {"real": [0, 0], "fake": [0, 0]}   # [correct, total]
    print(f"\n{'video':58s} {'true':5s} {'pred':5s} {'P(fake)':>8s}  face  ok")
    for path, true in jobs:
        try:
            r = run(det, path, a.frames)
        except Exception as e:
            print(f"{path.name[:58]:58s} ERROR: {e}")
            continue
        pred = r["label"].lower()
        ok = "" if true is None else ("OK" if pred == true else "WRONG")
        if true:
            tp[true][1] += 1
            tp[true][0] += pred == true
        print(f"{path.name[:58]:58s} {str(true or '-'):5s} {pred:5s} {r['fake_probability']:8.3f}  "
              f"{'yes' if r['face_found'] else 'NO '}   {ok}  ({r['seconds']}s)")
        rows.append(dict(video=path.name, true=true, pred=pred, fake_probability=r["fake_probability"],
                         face_found=r["face_found"], frames=r["frames_analyzed"]))

    total = sum(v[1] for v in tp.values())
    if total:
        print("\n=== SUMMARY ===")
        for k, (c, n) in tp.items():
            if n:
                print(f"{k.upper():5s} videos: {c}/{n} correct ({100 * c / n:.0f}%)")
        c = sum(v[0] for v in tp.values())
        print(f"OVERALL: {c}/{total} = {100 * c / total:.1f}%")

    out = Path(a.ckpt).parent / "video_test_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()