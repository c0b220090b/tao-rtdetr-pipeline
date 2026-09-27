#!/usr/bin/env bash
# お試し用: COCO val2017（画像 5000 枚・約 1GB）とアノテーションを data/raw/coco/ にダウンロードする
# 自分のデータが無いときに、パイプライン全体が動くかを確認するためのもの。
#
# ダウンロード後の変換例（交通系 4 クラス・最大 3000 枚）:
#   python3 scripts/03_prepare_dataset.py --classes person,bicycle,car,motorcycle --max-images 3000 \
#     coco --json data/raw/coco/annotations/instances_val2017.json --images data/raw/coco/val2017
source "$(dirname "$0")/common.sh"

DEST="$REPO_DIR/data/raw/coco"
mkdir -p "$DEST"
cd "$DEST" || exit 1

if [[ ! -d val2017 ]]; then
  info "画像をダウンロード（約 1GB）"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -q val2017.zip && rm val2017.zip
fi

if [[ ! -f annotations/instances_val2017.json ]]; then
  info "アノテーションをダウンロード（約 250MB）"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q annotations_trainval2017.zip annotations/instances_val2017.json && rm annotations_trainval2017.zip
fi

ok "完了: $DEST"
ls -la "$DEST" "$DEST/annotations"
