"""Kaggle GPU pipeline: ACDC rain -> features.csv + per_image_metrics.csv + real_rain_proxy.csv + examples.zip

Không chạy file này trực tiếp trên Kaggle: `python kaggle/build.py` nhúng package features/ (Member 2)
vào kaggle/kernel/run.py rồi push.

Nhãn Health (ACDC bản Kaggle không có GT box):
  1. Ảnh trời quang rain/*_ref (resize 1280x720) = clean. Pseudo-GT = YOLO11x (conf 0.4) trên clean.
  2. Degradation pixel-aligned: rain L1-3, blur L1-4, noise L1-3, dark L1-2, overexposure L1-2.
  3. YOLO11s chạy trên clean + degraded, matching IoU>=0.5 cùng class -> tp/fp/fn/recall.
     health_model/make_labels.py: health = recall / recall_clean.
  4. Ảnh mưa thật: không có clean khớp pixel -> features + proxy_recall (YOLO11s vs YOLO11x).

Degradation + detector ở đây là bản tạm (phần Member 1 / Member 4), giữ đúng schema output.
"""
import glob
import json
import os
import re
import shutil
import time
import zlib
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from features.extractor import extract

W, H = 1280, 720
DET_CLASSES = [0, 1, 2, 3, 5, 7]            # COCO: person, bicycle, car, motorcycle, bus, truck
MIN_GT_AREA = 24 * 24                       # bỏ pseudo-GT quá nhỏ (ở 1280x720)
N_EXAMPLES, N_REAL_EXAMPLES = 6, 12
OUT = "/kaggle/working"
EX_DIR = f"{OUT}/examples"
VARIANTS = ([("none", 0)] + [("rain", k) for k in (1, 2, 3)] + [("blur", k) for k in (1, 2, 3, 4)]
            + [("noise", k) for k in (1, 2, 3)] + [("dark", k) for k in (1, 2)]
            + [("overexposure", k) for k in (1, 2)])


# ---------------- degradation (Member 1 — bản tạm) ----------------
def synth_rain(img, k, rng):
    n, length, alpha, haze, blur = ([800, 2000, 4000][k - 1], [15, 25, 40][k - 1],
                                    [0.5, 0.65, 0.8][k - 1], [0.08, 0.16, 0.25][k - 1],
                                    [0.6, 1.0, 1.6][k - 1])
    out = img.astype(np.float32) * (1 - haze) + 200 * haze          # màn nước / haze
    out = cv2.GaussianBlur(out, (0, 0), blur)                        # giọt nước trên kính
    layer = np.zeros((H, W), np.float32)
    ang = np.deg2rad(rng.uniform(-15, 15))
    dx, dy = int(length * np.sin(ang)), int(length * np.cos(ang))
    for x, y, v in zip(rng.integers(0, W, n), rng.integers(0, H, n), rng.uniform(0.6, 1, n)):
        cv2.line(layer, (int(x), int(y)), (int(x) + dx, int(y) + dy), float(v), 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)[..., None] * alpha
    return np.clip(out * (1 - layer) + 230 * layer, 0, 255).astype(np.uint8)


def degrade(img, kind, k, rng):
    if kind == "none":
        return img
    if kind == "rain":
        return synth_rain(img, k, rng)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), [1.0, 2.0, 3.5, 6.0][k - 1])
    if kind == "noise":
        return np.clip(img + rng.normal(0, [10, 25, 45][k - 1], img.shape), 0, 255).astype(np.uint8)
    f = img.astype(np.float32) / 255
    if kind == "dark":
        f = f ** [1.3, 1.6][k - 1] * [0.35, 0.15][k - 1]
    elif kind == "overexposure":
        f = f * [1.8, 2.8][k - 1] + [0.08, 0.2][k - 1]
    return np.clip(f * 255, 0, 255).astype(np.uint8)


# ---------------- data ----------------
def find_rain_root():
    for pat in ["/kaggle/input/*/rgb_anon/rain", "/kaggle/input/*/*/rgb_anon/rain",
                "/kaggle/input/*/*/*/rgb_anon/rain", "/kaggle/input/*/*/*/*/rgb_anon/rain"]:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError("không thấy rgb_anon/rain trong /kaggle/input")


def frame_key(path):
    return re.match(r"(.*?_frame_\d+)", os.path.basename(path)).group(1)


def worker(task):
    path, kind, split, key = task
    img = cv2.resize(cv2.imread(path), (W, H), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    rows, imgs = [], []
    if kind == "ref":
        for t, k in VARIANTS:
            im = degrade(img, t, k, rng)
            iid = f"{key}_ref" if t == "none" else f"{key}_{t}{k}"
            rows.append({"image_id": iid, "parent_id": f"{key}_ref", "split": split,
                         "condition": "clear", "degradation_type": t, "degradation_level": k,
                         "is_synthetic": int(t != "none"), **extract(im)})
            imgs.append(im)
    else:
        rows.append({"image_id": f"{key}_rain", "parent_id": f"{key}_ref", "split": split,
                     "condition": "rain", "degradation_type": "real_rain", "degradation_level": 0,
                     "is_synthetic": 0, **extract(img)})
        imgs.append(img)
    return rows, imgs


# ---------------- detection (Member 4 — bản tạm) ----------------
def boxes(res, min_area=0):
    b = res.boxes
    xyxy, cls = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int)
    conf = b.conf.cpu().numpy()
    keep = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) >= min_area
    return xyxy[keep], cls[keep], conf[keep]


def iou(a, b):
    tl, br = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt, pred, thr=0.5):
    """Greedy theo confidence, cùng class. Trả về (gt_matched[bool], pred_tp[bool])."""
    gb, gc, _ = gt
    pb, pc, pconf = pred
    used, ptp = np.zeros(len(gb), bool), np.zeros(len(pb), bool)
    if len(gb) and len(pb):
        m = iou(pb, gb) * (pc[:, None] == gc[None, :])
        for i in np.argsort(-pconf):
            cand = np.where((m[i] >= thr) & ~used)[0]
            if len(cand):
                used[cand[np.argmax(m[i, cand])]] = ptp[i] = True
    return used, ptp


def save_example(rows, imgs, gt, preds, matches):
    key = rows[0]["parent_id"]
    d = f"{EX_DIR}/{key}"
    os.makedirs(d, exist_ok=True)
    info = {"gt": {"boxes": gt[0].round(1).tolist(), "cls": gt[1].tolist()}, "variants": {}}
    for row, im, pred, (gm, ptp) in zip(rows, imgs, preds, matches):
        cv2.imwrite(f"{d}/{row['image_id']}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        info["variants"][row["image_id"]] = {
            "type": row["degradation_type"], "level": row["degradation_level"],
            "gt_matched": gm.tolist(), "pred_boxes": pred[0].round(1).tolist(),
            "pred_cls": pred[1].tolist(), "pred_tp": ptp.tolist()}
    json.dump(info, open(f"{d}/boxes.json", "w"))


def main():
    t0 = time.time()
    root = find_rain_root()
    tasks = []
    for split in ("train", "val", "test"):
        for kind, sub in (("ref", f"{split}_ref"), ("rain", split)):
            files = sorted(glob.glob(os.path.join(root, sub, "*", "*.png")))
            print(f"{sub}: {len(files)} files  e.g. {os.path.basename(files[0]) if files else '-'}")
            tasks += [(f, kind, split, frame_key(f)) for f in files]

    det_s, det_x = YOLO("yolo11s.pt"), YOLO("yolo11x.pt")
    kw = dict(imgsz=W, classes=DET_CLASSES, verbose=False, half=True)
    feats, perf, proxy = [], [], []
    n_ex = n_real_ex = 0
    os.makedirs(f"{EX_DIR}/real_rain", exist_ok=True)
    with Pool(os.cpu_count()) as pool:
        for n, (rows, imgs) in enumerate(pool.imap(worker, tasks, chunksize=2), 1):
            gt = boxes(det_x.predict(imgs[0], conf=0.4, **kw)[0], MIN_GT_AREA)
            preds = [boxes(r) for r in det_s.predict(imgs, conf=0.25, batch=len(imgs), **kw)]
            matches = [match(gt, p) for p in preds]
            n_gt = len(gt[0])
            for row, pred, (gm, ptp) in zip(rows, preds, matches):
                tp, fp = int(gm.sum()), int((~ptp).sum())
                if row["degradation_type"] == "real_rain":
                    proxy.append({"image_id": row["image_id"], "n_ref": n_gt, "tp": tp, "fp": fp,
                                  "proxy_recall": tp / n_gt if n_gt else np.nan})
                else:
                    perf.append({"image_id": row["image_id"], "n_gt": n_gt, "tp": tp, "fp": fp,
                                 "fn": n_gt - tp, "precision": tp / max(tp + fp, 1),
                                 "recall": tp / n_gt if n_gt else np.nan})
            feats += rows
            if rows[0]["split"] == "test" and len(rows) > 1 and n_gt >= 6 and n_ex < N_EXAMPLES:
                save_example(rows, imgs, gt, preds, matches)
                n_ex += 1
            if rows[0]["degradation_type"] == "real_rain" and rows[0]["split"] == "test" \
                    and n_real_ex < N_REAL_EXAMPLES:
                cv2.imwrite(f"{EX_DIR}/real_rain/{rows[0]['image_id']}.jpg",
                            cv2.resize(imgs[0], (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 88])
                n_real_ex += 1
            if n % 100 == 0:
                print(f"{n}/{len(tasks)} images  {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(feats).round(6).to_csv(f"{OUT}/features.csv", index=False)
    pd.DataFrame(perf).round(4).to_csv(f"{OUT}/per_image_metrics.csv", index=False)
    pd.DataFrame(proxy).round(4).to_csv(f"{OUT}/real_rain_proxy.csv", index=False)
    shutil.make_archive(f"{OUT}/examples", "zip", EX_DIR)
    shutil.rmtree(EX_DIR)
    print(f"done: {len(feats)} feature rows, {len(perf)} perf rows, {len(proxy)} real-rain rows "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
