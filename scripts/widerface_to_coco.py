#!/usr/bin/env python3
"""
WIDER FACE のアノテーション（wider_face_*_bbx_gt.txt）を COCO 形式の JSON に変換する。
変換後は 03_prepare_dataset.py の coco モードで TAO 用に整える。

使い方:
  python3 scripts/widerface_to_coco.py --root data/raw/widerface
  → data/raw/widerface/train_coco.json, val_coco.json ができる

WIDER FACE の形式（1 画像ごと）:
  0--Parade/0_Parade_marchingband_1_849.jpg     ← 画像の相対パス
  1                                             ← 顔の数
  449 330 122 149 0 0 0 0 0 0                   ← x y w h blur expression illumination invalid occlusion pose
  （顔の数が 0 のときも "0 0 0 0 0 0 0 0 0 0" の 1 行が入る）

フィルタ:
  --min-size   これより小さい顔（幅か高さ、ピクセル）は除外。640x640 に縮小すると見えなくなる
               ような極小の顔を学習させると精度が落ちやすい。既定 10px
  --max-faces  顔の数（フィルタ後）がこれより多い画像は丸ごと除外。群衆写真は数百個の顔があり、
               RT-DETR の num_queries（既定 300）を超えると学習が不安定になる。既定 100
  invalid=1 の顔は常に除外。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow が必要です: pip install pillow")
    with Image.open(path) as im:
        return im.size


def parse_gt(gt_file: Path):
    """(相対パス, [(x, y, w, h, invalid), ...]) を順に返す"""
    lines = [l.strip() for l in gt_file.read_text().splitlines()]
    i = 0
    while i < len(lines):
        name = lines[i]
        if not name:
            i += 1
            continue
        n = int(lines[i + 1])
        i += 2
        rows = max(n, 1)  # 顔 0 個でもダミー行が 1 行ある
        faces = []
        for line in lines[i:i + rows][:n]:
            v = line.split()
            x, y, w, h = (float(t) for t in v[:4])
            invalid = int(v[7]) if len(v) > 7 else 0
            faces.append((x, y, w, h, invalid))
        i += rows
        yield name, faces


def convert(gt_file: Path, image_root: Path, out_json: Path, min_size: float, max_faces: int):
    images, annotations = [], []
    stats = dict(images=0, faces=0, small=0, invalid=0, crowded_images=0, no_face_images=0, missing=0)
    ann_id = 1
    for img_id, (rel, faces) in enumerate(parse_gt(gt_file), start=1):
        p = image_root / rel
        if not p.exists():
            stats["missing"] += 1
            continue
        keep = []
        for x, y, w, h, invalid in faces:
            if invalid:
                stats["invalid"] += 1
            elif w < min_size or h < min_size:
                stats["small"] += 1
            else:
                keep.append((x, y, w, h))
        if len(keep) > max_faces:
            stats["crowded_images"] += 1
            continue
        if not keep:
            # 顔が 0 個（または全部除外）の画像は使わない。小さい顔が写っているのに
            # 「顔なし」として学習すると、検出を抑える方向に学習してしまうため
            stats["no_face_images"] += 1
            continue
        w_img, h_img = image_size(p)
        images.append({"id": img_id, "file_name": rel, "width": w_img, "height": h_img})
        for x, y, w, h in keep:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        stats["images"] += 1
        stats["faces"] += len(keep)

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "face", "supercategory": "face"}]}
    out_json.write_text(json.dumps(coco))
    print(f"[{out_json.name}] 画像 {stats['images']} 枚 / 顔 {stats['faces']} 個")
    print(f"    除外: 小さい顔 {stats['small']} 個, invalid {stats['invalid']} 個, "
          f"顔が多すぎる画像 {stats['crowded_images']} 枚, 顔が残らない画像 {stats['no_face_images']} 枚"
          + (f", 画像が無い {stats['missing']} 件" if stats["missing"] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw/widerface",
                    help="WIDER_train/, WIDER_val/, wider_face_split/ があるフォルダ")
    ap.add_argument("--min-size", type=float, default=10)
    ap.add_argument("--max-faces", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.root)
    split_dir = root / "wider_face_split"
    for split, folder in (("train", "WIDER_train"), ("val", "WIDER_val")):
        gt = split_dir / f"wider_face_{split}_bbx_gt.txt"
        img_root = root / folder / "images"
        if not gt.exists() or not img_root.is_dir():
            sys.exit(f"{gt} または {img_root} がありません。先に sample_download_widerface.sh を実行してください")
        convert(gt, img_root, root / f"{split}_coco.json", args.min_size, args.max_faces)

    print("\n次に TAO 用に整える:")
    print(f"  python3 scripts/03_prepare_dataset.py coco \\\n"
          f"    --json {root}/train_coco.json --images {root}/WIDER_train/images \\\n"
          f"    --val-json {root}/val_coco.json --val-images {root}/WIDER_val/images")


if __name__ == "__main__":
    main()
