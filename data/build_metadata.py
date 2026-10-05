"""[Member 1 - Ngô Xuân Hoàng] Gom thông tin mọi ảnh thành dataset/metadata.csv.

Data Contract (README.md):
  image_id, file_path, parent_id, scene_id, camera, condition,
  degradation_type, degradation_level, day_night, weather, split
"""
import csv
import os
import re
import sys
from pathlib import Path

from common.config import load_config

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def extract_scene_id(image_id: str) -> str:
    m = re.match(r"(GOPR\d+|GP\d+|[A-Za-z0-9]+)_frame_", image_id)
    if m:
        return m.group(1)
    parts = image_id.split("_")
    return parts[0] if parts else "scene_001"


def build_metadata(features_path: Path, output_path: Path) -> int:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found at: {features_path}")

    print(f"Reading features from: {features_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "file_path",
        "parent_id",
        "scene_id",
        "camera",
        "condition",
        "degradation_type",
        "degradation_level",
        "day_night",
        "weather",
        "split",
    ]

    count = 0
    with open(features_path, mode="r", encoding="utf-8") as f_in, \
         open(output_path, mode="w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            iid = row["image_id"]
            pid = row["parent_id"]
            split = row["split"]
            deg_type = row["degradation_type"]
            deg_lvl = row["degradation_level"]
            cond = row["condition"]

            scene_id = extract_scene_id(iid)
            camera = "CAM_FRONT"
            day_night = "day"
            weather = "rain" if deg_type in ("rain", "real_rain") or cond == "rain" else "clear"

            if deg_type == "none":
                file_path = f"dataset/{split}/{iid}.jpg"
            elif deg_type == "real_rain":
                file_path = f"dataset/{split}/real_rain/{iid}.jpg"
            else:
                file_path = f"dataset/corrupted/{deg_type}/L{deg_lvl}/{iid}.jpg"

            writer.writerow({
                "image_id": iid,
                "file_path": file_path,
                "parent_id": pid,
                "scene_id": scene_id,
                "camera": camera,
                "condition": cond,
                "degradation_type": deg_type,
                "degradation_level": deg_lvl,
                "day_night": day_night,
                "weather": weather,
                "split": split,
            })
            count += 1

    print(f"Successfully generated {count} metadata rows -> {output_path}")
    return count


def main():
    cfg = load_config()
    feat_path = Path(cfg["paths"]["features"])
    meta_path = Path(cfg["paths"]["metadata"])
    build_metadata(feat_path, meta_path)


if __name__ == "__main__":
    main()
