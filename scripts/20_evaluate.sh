#!/usr/bin/env bash
# val データで精度（mAP）を評価する
#   ./scripts/20_evaluate.sh                               # 最新のチェックポイント
#   ./scripts/20_evaluate.sh /workspace/results/.../model_epoch_050.pth
source "$(dirname "$0")/common.sh"

ckpt_arg=""
if [[ "${1:-}" == *.pth ]]; then ckpt_arg="$1"; shift; fi
ckpt="$(find_checkpoint "$ckpt_arg")"
info "評価するチェックポイント: $ckpt"

# shellcheck disable=SC2046
tao_run rtdetr evaluate -e "$SPEC" $(common_overrides) \
  "evaluate.checkpoint=$ckpt" \
  "evaluate.input_width=$INPUT_WIDTH" "evaluate.input_height=$INPUT_HEIGHT" \
  "$@"

ok "結果: $RESULTS_HOST/evaluate"
