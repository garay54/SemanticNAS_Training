#!/usr/bin/env python3
"""
Convierte etiquetas YOLO-seg polygon a mascaras PNG semanticas.

YOLO usa class_id empezando en 0 para las clases de objeto. Este script genera:
  0 = fondo
  class_id + --class-offset = clase semantica

Entrada esperada:
  yolo_dataset/{train,valid,test}/images
  yolo_dataset/{train,valid,test}/labels

Salida:
  output/{train,valid,test}/images
  output/{train,valid,test}/masks
  output/classes.json
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_yolo_names(data_yaml):
    if not data_yaml:
        return []
    path = Path(data_yaml)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        def sort_key(key):
            text = str(key)
            return int(text) if text.isdigit() else text
        return [str(names[k]) for k in sorted(names, key=sort_key)]
    return [str(x) for x in names]


def yolo_polygon_to_mask(label_path, width, height, class_offset, fill_order):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask

    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            points = np.asarray(
                [[coords[i] * width, coords[i + 1] * height] for i in range(0, len(coords), 2)],
                dtype=np.float32,
            )
            if len(points) >= 3:
                rows.append((class_id, points))

    if fill_order == "large_first":
        rows.sort(key=lambda item: abs(cv2.contourArea(item[1].astype(np.float32))), reverse=True)
    elif fill_order == "small_first":
        rows.sort(key=lambda item: abs(cv2.contourArea(item[1].astype(np.float32))))

    for class_id, points in rows:
        pts = np.round(points).astype(np.int32)
        semantic_id = class_id + class_offset
        cv2.fillPoly(mask, [pts], int(semantic_id))
    return mask


def collect_images(image_dir):
    images = []
    for ext in IMAGE_EXTS:
        images.extend(glob.glob(str(image_dir / f"*{ext}")))
    return sorted(images)


def write_classes_json(output_dir, names, class_offset):
    classes = [{"id": 0, "name": "background", "color": [0, 0, 0]}]
    colors = [
        [0, 255, 0],
        [255, 128, 0],
        [0, 128, 255],
        [255, 0, 128],
        [180, 255, 0],
        [160, 80, 255],
    ]
    for idx, name in enumerate(names):
        semantic_id = idx + class_offset
        color = colors[idx % len(colors)]
        classes.append({"id": semantic_id, "name": name, "color": color})
    out = {"classes": classes}
    (output_dir / "classes.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def process_split(args, split):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    img_in = input_dir / split / "images"
    lbl_in = input_dir / split / "labels"
    img_out = output_dir / split / "images"
    mask_out = output_dir / split / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    images = collect_images(img_in)
    ok = 0
    for image_path_str in images:
        image_path = Path(image_path_str)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [WARN] No se pudo leer {image_path}")
            continue
        h, w = image.shape[:2]
        label_path = lbl_in / f"{image_path.stem}.txt"
        mask = yolo_polygon_to_mask(label_path, w, h, args.class_offset, args.fill_order)

        if args.copy_mode == "copy":
            shutil.copy2(image_path, img_out / image_path.name)
        elif args.copy_mode == "hardlink":
            target = img_out / image_path.name
            if target.exists():
                target.unlink()
            try:
                target.hardlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, target)
        else:
            raise ValueError(f"copy_mode no soportado: {args.copy_mode}")

        cv2.imwrite(str(mask_out / f"{image_path.stem}.png"), mask)
        ok += 1
    return ok


def parse_args():
    parser = argparse.ArgumentParser(description="Convierte YOLO polygon a mascaras semanticas.")
    parser.add_argument("--input", required=True, help="Dataset YOLO-seg.")
    parser.add_argument("--output", required=True, help="Dataset images/masks de salida.")
    parser.add_argument("--data-yaml", default="", help="data.yaml opcional para nombres de clase.")
    parser.add_argument("--class-offset", type=int, default=1, help="YOLO class_id + offset. Default: 1.")
    parser.add_argument("--fill-order", choices=["file", "large_first", "small_first"], default="file")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = load_yolo_names(args.data_yaml)
    write_classes_json(output_dir, names, args.class_offset)

    print("\n" + "=" * 60)
    print(f"YOLO polygons -> mascaras semanticas")
    print(f"Entrada: {args.input}")
    print(f"Salida:  {args.output}")
    print("=" * 60 + "\n")

    totals = {}
    for split in SPLITS:
        n = process_split(args, split)
        totals[split] = n
        print(f"  {split:>5}: {n:6d} imagenes")

    print("\n" + "=" * 60)
    print(f"TOTAL: {sum(totals.values())} mascaras")
    print(f"Clases: {output_dir / 'classes.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
