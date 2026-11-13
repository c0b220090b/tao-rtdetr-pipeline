#!/usr/bin/env python3
"""
Isaac Sim（Replicator の KittiWriter）が書き出した KITTI 形式のラベルを、COCO 形式の JSON にまとめる。
変換後は 03_prepare_dataset.py の coco モードで TAO 用に整える。

使い方:
  python3 scripts/kitti_to_coco.py --root data/raw/palletjack_sdg --out data/raw/palletjack_sdg/synth_coco.json
  （--root の下の各サブフォルダ（distractors_warehouse など）を 1 つの出力として扱い、全部まとめる）

KITTI のラベル（1 行 1 物体、空白区切り）:
  クラス名 truncated occluded alpha x1 y1 x2 y2 ...
  ここでは クラス名 と x1 y1 x2 y2（ピクセル）だけ使う。

画像はラベルと同じファイル名（拡張子違い）のものを、同じサブフォルダの中から探す。
COCO の file_name は --root からの相対パスになる（03_prepare_dataset.py の --images に --root を渡す）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow が必要です（TAO コンテナの中で実行すると入っています）")
    with Image.open(path) as im:
        return im.size


def parse_kitti(txt: Path):
    boxes = []
    for line in txt.read_text().splitlines():
        v = line.split()
        if len(v) < 8:
            continue
        try:
            x1, y1, x2, y2 = (float(t) for t in v[4:8])
        except ValueError:
            continue
        boxes.append((v[0], x1, y1, x2, y2))
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="生成データのフォルダ（サブフォルダごとに 1 回分の出力）")
    ap.add_argument("--out", required=True, help="出力する COCO JSON")
    ap.add_argument("--class-name", default="palletjack", help="COCO に書くクラス名（KITTI のクラス名がこれと違っても統一する）")
    ap.add_argument("--keep-class", default=None,
                    help="KITTI のこのクラス名だけ使う（既定: 全部を --class-name として扱う）")
    ap.add_argument("--min-size", type=float, default=8, help="幅か高さがこれ未満の枠は捨てる（px）")
    ap.add_argument("--keep-empty", action="store_true",
                    help="対象が写っていない画像も残す（既定は捨てる）")
    args = ap.parse_args()

    root = Path(args.root)
    subdirs = sorted(p for p in root.iterdir() if p.is_dir()) or [root]
    images, annotations = [], []
    stats = Counter()
    seen_classes = Counter()
    ann_id = img_id = 1

    for sub in subdirs:
        # 画像をファイル名（拡張子なし）で引けるようにする
        index = {}
        for p in sub.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and "semantic" not in str(p).lower():
                index.setdefault(p.stem, p)
        labels = [t for t in sub.rglob("*.txt") if t.stem in index]
        if not labels:
            print(f"[WARN] {sub}: ラベルと画像の組が見つかりません")
            continue
        n_before = len(images)
        for txt in sorted(labels):
            img = index[txt.stem]
            boxes = parse_kitti(txt)
            keep = []
            for cls, x1, y1, x2, y2 in boxes:
                seen_classes[cls] += 1
                if args.keep_class and cls != args.keep_class:
                    continue
                w, h = x2 - x1, y2 - y1
                if w < args.min_size or h < args.min_size:
                    stats["small"] += 1
                    continue
                keep.append((x1, y1, w, h))
            if not keep and not args.keep_empty:
                stats["empty_images"] += 1
                continue
            W, H = image_size(img)
            images.append({"id": img_id, "file_name": str(img.relative_to(root)), "width": W, "height": H})
            for x, y, w, h in keep:
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
                ann_id += 1
            img_id += 1
        print(f"[{sub.name}] 画像 {len(images) - n_before} 枚")

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": args.class_name, "supercategory": "object"}]}
    Path(args.out).write_text(json.dumps(coco))
    print(f"\n合計: 画像 {len(images)} 枚 / 枠 {len(annotations)} 個 → {args.out}")
    print(f"除外: 小さい枠 {stats['small']} 個, 対象が写っていない画像 {stats['empty_images']} 枚")
    print(f"KITTI に出てきたクラス名: {dict(seen_classes)}")


if __name__ == "__main__":
    main()
