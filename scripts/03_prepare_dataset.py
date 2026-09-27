#!/usr/bin/env python3
"""
データセットを TAO RT-DETR 用の COCO 形式に整える。

出力（data/processed/）:
    images/train/, images/val/        画像（ハードリンク。別ドライブならコピー）
    annotations/train.json, val.json  COCO 形式のアノテーション
    classmap.txt                      クラス名（1 行 1 クラス、ID 順）
    num_classes.txt                   spec の dataset.num_classes に入れる値（クラス数 + 1）

入力は 3 通り:
  (A) COCO 形式 1 つ（train/val に自動分割）
      python3 scripts/03_prepare_dataset.py coco --json anno.json --images imgs/
  (B) COCO 形式で train / val が分かれている
      python3 scripts/03_prepare_dataset.py coco --json train.json --images train_imgs/ \\
          --val-json val.json --val-images val_imgs/
  (C) YOLO 形式（images/ と labels/、クラス名は classes.txt か data.yaml）
      python3 scripts/03_prepare_dataset.py yolo --root yolo_dataset/ --names classes.txt

なぜ ID を振り直すのか:
  TAO の RT-DETR は、カスタムデータ（remap_mscoco_category: false）では COCO の
  category_id をそのままクラス番号として使う。そのため num_classes は「最大の ID + 1」
  以上が必要になる。ここでは ID を 1..N に詰め直し、num_classes = N + 1 とする
  （0 番は未使用の "background"。書き出し時の labels.txt も 0 番が background になる）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPO_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------
def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow が必要です: pip install pillow")
    with Image.open(path) as im:
        return im.size  # (width, height)


def place_image(src: Path, dst: Path) -> None:
    """ハードリンク（同じドライブなら容量を使わない）→ 失敗したらコピー。
    シンボリックリンクは Docker のマウント外を指すと読めなくなるので使わない。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clip_box(x, y, w, h, iw, ih):
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(iw), x + w), min(float(ih), y + h)
    return x1, y1, x2 - x1, y2 - y1


class Builder:
    """画像とボックスを受け取り、ID を詰め直した COCO JSON を作る。"""

    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self.categories = [{"id": i + 1, "name": n, "supercategory": "object"}
                           for i, n in enumerate(class_names)]
        self.splits = {"train": {"images": [], "annotations": []},
                       "val": {"images": [], "annotations": []}}
        self.used_names: dict[str, set] = {"train": set(), "val": set()}
        self.ann_id = 1
        self.img_id = 1
        self.dropped = 0
        self.clipped = 0

    def add(self, split: str, src: Path, width: int, height: int, boxes, out_dir: Path):
        """boxes: [(class_index_0based, x, y, w, h)]（ピクセル、左上 + 幅高さ）"""
        # ファイル名の衝突を避ける（別フォルダに同名ファイルがある場合）
        name = src.name
        stem, ext = src.stem, src.suffix
        n = 1
        while name in self.used_names[split]:
            name = f"{stem}_{n}{ext}"
            n += 1
        self.used_names[split].add(name)

        place_image(src, out_dir / "images" / split / name)
        img_id = self.img_id
        self.img_id += 1
        self.splits[split]["images"].append(
            {"id": img_id, "file_name": name, "width": width, "height": height})

        for cls, x, y, w, h in boxes:
            nx, ny, nw, nh = clip_box(x, y, w, h, width, height)
            if nw < 1 or nh < 1:
                self.dropped += 1
                continue
            if (nx, ny, nw, nh) != (x, y, w, h):
                self.clipped += 1
            self.splits[split]["annotations"].append({
                "id": self.ann_id, "image_id": img_id, "category_id": cls + 1,
                "bbox": [round(nx, 2), round(ny, 2), round(nw, 2), round(nh, 2)],
                "area": round(nw * nh, 2), "iscrowd": 0,
            })
            self.ann_id += 1

    def write(self, out_dir: Path):
        ann_dir = out_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        for split, data in self.splits.items():
            coco = {"images": data["images"], "annotations": data["annotations"],
                    "categories": self.categories}
            (ann_dir / f"{split}.json").write_text(json.dumps(coco, ensure_ascii=False))
        (out_dir / "classmap.txt").write_text("\n".join(self.class_names) + "\n")
        (out_dir / "num_classes.txt").write_text(f"{len(self.class_names) + 1}\n")

    def report(self):
        print("\n==== 結果 ====")
        print(f"クラス数: {len(self.class_names)}  →  dataset.num_classes = {len(self.class_names) + 1}")
        for split, data in self.splits.items():
            cnt = Counter(a["category_id"] for a in data["annotations"])
            empty = len({i["id"] for i in data["images"]} -
                        {a["image_id"] for a in data["annotations"]})
            print(f"\n[{split}] 画像 {len(data['images'])} 枚 / ボックス {len(data['annotations'])} 個"
                  f"（ボックスなしの画像 {empty} 枚）")
            for c in self.categories:
                print(f"  {c['id']:3d} {c['name']:<20s} {cnt.get(c['id'], 0):7d}")
        if self.clipped:
            print(f"\n画像外にはみ出たボックスを {self.clipped} 個、画像内に切り詰めました")
        if self.dropped:
            print(f"幅か高さが 1px 未満のボックスを {self.dropped} 個、除外しました")
        if not self.splits["val"]["images"]:
            print("\n[WARN] val が 0 枚です。--val-ratio を確認してください")


def split_names(items: list, val_ratio: float, seed: int) -> set:
    rng = random.Random(seed)
    items = sorted(items)
    rng.shuffle(items)
    n_val = max(1, round(len(items) * val_ratio)) if val_ratio > 0 and len(items) > 1 else 0
    return set(items[:n_val])


# ---------------------------------------------------------------------------
# COCO 入力
# ---------------------------------------------------------------------------
def load_coco(json_path: Path, image_root: Path):
    coco = json.loads(json_path.read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    by_img: dict[int, list] = {}
    for a in coco.get("annotations", []):
        if a.get("iscrowd", 0):
            continue
        by_img.setdefault(a["image_id"], []).append(a)
    images = []
    missing = 0
    for im in coco["images"]:
        p = image_root / im["file_name"]
        if not p.exists():
            missing += 1
            continue
        images.append((im, p, by_img.get(im["id"], [])))
    if missing:
        print(f"[WARN] {json_path.name}: 画像ファイルが見つからないものを {missing} 件スキップ")
    return cats, images


def run_coco(args, out_dir: Path):
    cats, train_imgs = load_coco(Path(args.json), Path(args.images))
    val_imgs = None
    if args.val_json:
        vcats, val_imgs = load_coco(Path(args.val_json), Path(args.val_images))
        if [c["name"] for c in vcats] != [c["name"] for c in cats]:
            sys.exit("train と val でクラス（categories）が一致しません")

    names = [c["name"] for c in cats]
    old2new = {c["id"]: i for i, c in enumerate(cats)}  # 0 始まりに（Builder が +1 する）
    b = Builder(names)

    def boxes_of(anns):
        return [(old2new[a["category_id"]], *a["bbox"]) for a in anns if a["category_id"] in old2new]

    def dims(im, p):
        w, h = im.get("width"), im.get("height")
        return (w, h) if w and h else image_size(p)

    if val_imgs is None:
        val_set = split_names([str(p) for _, p, _ in train_imgs], args.val_ratio, args.seed)
        for im, p, anns in train_imgs:
            b.add("val" if str(p) in val_set else "train", p, *dims(im, p), boxes_of(anns), out_dir)
    else:
        for im, p, anns in train_imgs:
            b.add("train", p, *dims(im, p), boxes_of(anns), out_dir)
        for im, p, anns in val_imgs:
            b.add("val", p, *dims(im, p), boxes_of(anns), out_dir)
    return b


# ---------------------------------------------------------------------------
# YOLO 入力
# ---------------------------------------------------------------------------
def read_names(path: Path) -> list[str]:
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            sys.exit("data.yaml を読むには PyYAML が必要です: pip install pyyaml")
        names = yaml.safe_load(path.read_text())["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return list(names)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def yolo_pairs(root: Path):
    """(split or None, 画像パス, ラベルパス) を列挙する。
    対応する構成: root/images/{train,val}/ + root/labels/{train,val}/  または
                   root/images/ + root/labels/（分割なし）"""
    img_root, lbl_root = root / "images", root / "labels"
    if not img_root.is_dir():
        sys.exit(f"{img_root} がありません")
    subsplits = [s for s in ("train", "val", "valid") if (img_root / s).is_dir()]
    groups = [(s, img_root / s, lbl_root / s) for s in subsplits] or [(None, img_root, lbl_root)]
    for split, idir, ldir in groups:
        for p in sorted(idir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                rel = p.relative_to(idir).with_suffix(".txt")
                split_name = "val" if split in ("val", "valid") else split
                yield split_name, p, ldir / rel


def run_yolo(args, out_dir: Path):
    names = read_names(Path(args.names))
    b = Builder(names)
    pairs = list(yolo_pairs(Path(args.root)))
    has_split = any(s is not None for s, _, _ in pairs)
    val_set = set() if has_split else split_names([str(p) for _, p, _ in pairs], args.val_ratio, args.seed)
    bad = 0
    for split, img, lbl in pairs:
        w, h = image_size(img)
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                if not 0 <= cls < len(names):
                    bad += 1
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
        target = split if has_split else ("val" if str(img) in val_set else "train")
        b.add(target, img, w, h, boxes, out_dir)
    if bad:
        print(f"[WARN] クラス番号が範囲外のボックスを {bad} 個スキップしました")
    return b


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_DIR / "data" / "processed"), help="出力先")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="val に回す割合（分割済みなら無視）")
    ap.add_argument("--seed", type=int, default=42)
    sub = ap.add_subparsers(dest="fmt", required=True)

    c = sub.add_parser("coco")
    c.add_argument("--json", required=True)
    c.add_argument("--images", required=True)
    c.add_argument("--val-json")
    c.add_argument("--val-images")

    y = sub.add_parser("yolo")
    y.add_argument("--root", required=True)
    y.add_argument("--names", required=True, help="classes.txt または data.yaml")

    args = ap.parse_args()
    if args.fmt == "coco" and bool(args.val_json) != bool(args.val_images):
        ap.error("--val-json と --val-images はセットで指定してください")

    out_dir = Path(args.out)
    for sub_dir in ("images", "annotations"):
        shutil.rmtree(out_dir / sub_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    b = run_coco(args, out_dir) if args.fmt == "coco" else run_yolo(args, out_dir)
    b.write(out_dir)
    b.report()
    print(f"\n出力先: {out_dir}")


if __name__ == "__main__":
    main()
