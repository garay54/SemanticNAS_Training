#!/usr/bin/env python3
"""
Tilea un dataset de segmentacion semantica preservando IDs de clase.

Entrada esperada:
  dataset/{train,valid,test}/images
  dataset/{train,valid,test}/masks

Salida:
  output/{train,valid,test}/images
  output/{train,valid,test}/masks

Las mascaras deben ser PNG de un canal con:
  0 = fondo
  1..N = clases de objeto
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np


SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_tiles(width, height, tile_size, overlap):
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("--overlap debe ser menor que --tile-size")

    def positions(size):
        if size <= tile_size:
            return [0]
        pos = list(range(0, size - tile_size, stride))
        if not pos or pos[-1] + tile_size < size:
            pos.append(size - tile_size)
        return sorted(set(pos))

    return [
        (x, y, x + tile_size, y + tile_size)
        for y in positions(height)
        for x in positions(width)
    ]


def read_mask(path, height, width):
    if not path.exists():
        return np.zeros((height, width), dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def foreground_fraction(mask, background_id, foreground_ids):
    if foreground_ids:
        fg = np.isin(mask, foreground_ids)
    else:
        fg = mask != background_id
    return float(fg.mean())


def process_split(args, split):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    img_in = input_dir / split / "images"
    mask_in = input_dir / split / "masks"
    img_out = output_dir / split / "images"
    mask_out = output_dir / split / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in IMAGE_EXTS:
        images.extend(glob.glob(str(img_in / f"*{ext}")))
    images = sorted(images)
    if not images:
        print(f"  [{split}] Sin imagenes, omitiendo.")
        return 0, 0

    foreground_ids = [int(x) for x in args.foreground_ids.split(",") if x.strip()] if args.foreground_ids else []
    total = 0
    total_fg = 0

    for image_path_str in images:
        image_path = Path(image_path_str)
        stem = image_path.stem
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [WARN] No se pudo leer {image_path}")
            continue
        h, w = image.shape[:2]
        mask = read_mask(mask_in / f"{stem}.png", h, w)

        for tile_idx, (x1, y1, x2, y2) in enumerate(get_tiles(w, h, args.tile_size, args.overlap)):
            tile_image = image[y1:y2, x1:x2]
            tile_mask = mask[y1:y2, x1:x2]

            if tile_image.shape[0] != args.tile_size or tile_image.shape[1] != args.tile_size:
                tile_image = cv2.resize(tile_image, (args.tile_size, args.tile_size), interpolation=cv2.INTER_LINEAR)
                tile_mask = cv2.resize(tile_mask, (args.tile_size, args.tile_size), interpolation=cv2.INTER_NEAREST)

            fg_frac = foreground_fraction(tile_mask, args.background_id, foreground_ids)
            if fg_frac < args.min_foreground:
                if args.keep_empty_every <= 0 or tile_idx % args.keep_empty_every != 0:
                    continue

            tile_name = f"{stem}_t{tile_idx:04d}"
            cv2.imwrite(str(img_out / f"{tile_name}.jpg"), tile_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(mask_out / f"{tile_name}.png"), tile_mask)
            total += 1
            if fg_frac >= args.min_foreground:
                total_fg += 1

    return total, total_fg


def parse_args():
    parser = argparse.ArgumentParser(description="Tilea datasets semanticos images/masks.")
    parser.add_argument("--input", required=True, help="Dataset de entrada.")
    parser.add_argument("--output", required=True, help="Dataset tileado de salida.")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--min-foreground", type=float, default=0.001)
    parser.add_argument("--keep-empty-every", type=int, default=5, help="Guarda 1 de cada N tiles vacios. 0 descarta todos.")
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument("--foreground-ids", default="", help="CSV opcional de clases foreground para filtrar, ej. 1,2,3")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 60)
    print(f"Tileado semantico: {args.input}")
    print(f"Salida:            {args.output}")
    print(f"Tile: {args.tile_size} | overlap: {args.overlap} | min_foreground: {args.min_foreground}")
    print("=" * 60 + "\n")

    totals = {}
    for split in SPLITS:
        n, nf = process_split(args, split)
        totals[split] = n
        print(f"  {split:>5}: {n:6d} tiles ({nf} con foreground)")

    print("\n" + "=" * 60)
    print(f"TOTAL: {sum(totals.values())} tiles")
    print(f"Dataset listo en: {os.path.abspath(args.output)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
