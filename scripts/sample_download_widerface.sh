#!/usr/bin/env bash
# 顔検出用: WIDER FACE（train 12,880 枚 / val 3,226 枚、約 1.8GB）を data/raw/widerface/ にダウンロードする
# 配布元: http://shuoyang1213.me/WIDERFACE/ （Hugging Face のミラーから取得）
# ライセンス: 研究・非商用目的のみ
#
# このあと:
#   python3 scripts/widerface_to_coco.py --root data/raw/widerface
source "$(dirname "$0")/common.sh"

DEST="$REPO_DIR/data/raw/widerface"
BASE=https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data
mkdir -p "$DEST"
cd "$DEST" || exit 1

for name in WIDER_train WIDER_val wider_face_split; do
  if [[ -d "$name" ]]; then
    info "$name は展開済み"
    continue
  fi
  info "ダウンロード: $name.zip"
  wget -c "$BASE/$name.zip" -O "$name.zip"
  unzip -q "$name.zip" && rm "$name.zip"
done

ok "完了: $DEST"
ls "$DEST" "$DEST/wider_face_split"
