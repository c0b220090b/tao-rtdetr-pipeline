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

# TAO は color_map に載っているクラスしか画像に枠を描かない（既定は COCO の 80 クラス）。
# classmap.txt のクラス名から color_map を作って渡す（自分で inference.color_map=... を指定したらそちらを使う）
color_arg=()
if [[ " $* " != *"inference.color_map="* ]]; then
  colors=(red lime blue yellow cyan magenta orange purple pink white)
  entries=() i=0
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if [[ "$name" =~ [[:space:],:{}] ]]; then
      warn "クラス名 '$name' に空白などが含まれるため、枠の色を自動設定できません（画像に描かれません）"
    else
      entries+=("$name:${colors[$(( i % ${#colors[@]} ))]}")
    fi
    i=$(( i + 1 ))
  done < "$REPO_DIR/data/processed/classmap.txt"
  if (( ${#entries[@]} )); then
    color_arg=("inference.color_map={$(IFS=,; echo "${entries[*]}")}")
    info "枠の色: ${color_arg[0]#inference.color_map=}"
  fi
fi

[[ -f "$REPO_DIR/data/processed/classmap_infer.txt" ]] || {
  error "data/processed/classmap_infer.txt がありません。scripts/03_prepare_dataset.py を実行し直してください"; exit 1; }

# shellcheck disable=SC2046
tao_run rtdetr inference -e "$SPEC" $(common_overrides) \
  "inference.checkpoint=$ckpt" \
  "inference.input_width=$INPUT_WIDTH" "inference.input_height=$INPUT_HEIGHT" \
  "${color_arg[@]}" \
  "$@"

ok "結果画像: $RESULTS_HOST/inference"
