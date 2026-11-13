#!/usr/bin/env python3
"""
LOCO データセット（実際の物流倉庫の写真）から、ハンドパレット（pallet_truck）の写っている画像だけを取り出し、
クラス名を palletjack にした COCO JSON を作る。合成データで訓練したモデルを「実画像で」評価するために使う。

LOCO: https://github.com/tum-fml/loco （パブリックドメイン）
  アノテーション: rgb/loco-all-v1.json（COCO 形式。クラス: small_load_carrier, forklift, pallet, stillage, pallet_truck）
  画像: README のリンクからダウンロードしたものを展開（サブフォルダの中に散らばっていてもよい）

使い方:
  python3 scripts/loco_to_coco.py --json data/raw/loco/rgb/loco-all-v1.json \
      --images data/raw/loco/images --out data/raw/loco/palletjack_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True, help="LOCO の画像を展開したフォルダ（下のフォルダも探す）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-class", default="pallet_truck")
    ap.add_argument("--dst-class", default="palletjack")
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text())
    src = [c["id"] for c in d["categories"] if c["name"] == args.src_class]
    if not src:
        sys.exit(f"{args.src_class} がありません。クラス: {[c['name'] for c in d['categories']]}")
    src_id = src[0]

    root = Path(args.images)
    found = {p.name: p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    anns_by_img = {}
    for a in d["annotations"]:
        if a["category_id"] == src_id:
            anns_by_img.setdefault(a["image_id"], []).append(a)

    images, annotations, missing = [], [], 0
    for im in d["images"]:
        if im["id"] not in anns_by_img:
            continue
        p = found.get(im["file_name"])
        if p is None:
            missing += 1
            continue
        images.append({"id": im["id"], "file_name": str(p.relative_to(root)),
                       "width": im["width"], "height": im["height"]})
        for a in anns_by_img[im["id"]]:
            annotations.append({"id": a["id"], "image_id": im["id"], "category_id": 1,
                                "bbox": a["bbox"], "area": a["bbox"][2] * a["bbox"][3], "iscrowd": 0})

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": args.dst_class, "supercategory": "object"}]}
    Path(args.out).write_text(json.dumps(coco))
    print(f"画像 {len(images)} 枚 / {args.dst_class} の枠 {len(annotations)} 個 → {args.out}")
    if missing:
        print(f"[WARN] JSON にあるが画像ファイルが見つからないもの: {missing} 枚（--images を確認）")


if __name__ == "__main__":
    main()
