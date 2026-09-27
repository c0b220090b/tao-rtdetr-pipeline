#!/usr/bin/env bash
# 画像に推論して、ボックスを描いた画像を保存する（目視確認用）
#   ./scripts/30_inference.sh                       # val 画像・最新チェックポイント
#   ./scripts/30_inference.sh <ckpt> inference.conf_threshold=0.3
# 別の画像フォルダで試すときは、リポジトリ内に置いて次を追加:
#   'dataset.infer_data_sources.image_dir=[/workspace/data/raw/test_images]'
source "$(dirname "$0")/common.sh"

ckpt_arg=""
if [[ "${1:-}" == *.pth ]]; then ckpt_arg="$1"; shift; fi
ckpt="$(find_checkpoint "$ckpt_arg")"
info "チェックポイント: $ckpt"

# shellcheck disable=SC2046
tao_run rtdetr inference -e "$SPEC" $(common_overrides) \
  "inference.checkpoint=$ckpt" \
  "inference.input_width=$INPUT_WIDTH" "inference.input_height=$INPUT_HEIGHT" \
  "$@"

ok "結果画像: $RESULTS_HOST/inference"
