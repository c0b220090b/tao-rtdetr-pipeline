#!/usr/bin/env bash
# ONNX に書き出す（Jetson に持っていくファイル一式を作る）
#   ./scripts/40_export.sh           # 最新のチェックポイント
#   ./scripts/40_export.sh <ckpt>
#
# 出力: results/<EXPERIMENT>/export/
#   model.onnx               … Jetson で TensorRT エンジンに変換する
#   labels.txt               … クラス名（0 番は background）
#   nvdsinfer_config.yaml    … TAO が生成する DeepStream 設定（参考）
source "$(dirname "$0")/common.sh"

ckpt_arg=""
if [[ "${1:-}" == *.pth ]]; then ckpt_arg="$1"; shift; fi
ckpt="$(find_checkpoint "$ckpt_arg")"

EXPORT_HOST="$RESULTS_HOST/export"
ONNX="$RESULTS_DIR/export/model.onnx"
if [[ -f "$EXPORT_HOST/model.onnx" ]]; then
  # TAO は既存ファイルがあるとエラーで止まるので、日時を付けて退避する
  mv "$EXPORT_HOST" "${EXPORT_HOST}_$(date +%Y%m%d-%H%M%S)"
  warn "以前の export フォルダを退避しました"
fi
info "書き出すチェックポイント: $ckpt"

# shellcheck disable=SC2046
tao_run rtdetr export -e "$SPEC" $(common_overrides) \
  "export.checkpoint=$ckpt" \
  "export.onnx_file=$ONNX" \
  "export.input_width=$INPUT_WIDTH" "export.input_height=$INPUT_HEIGHT" \
  "$@"

# Jetson に持っていく一式をまとめる
BUNDLE="$RESULTS_HOST/jetson_bundle_${EXPERIMENT}.tar.gz"
tar -czf "$BUNDLE" -C "$EXPORT_HOST" .
ok "書き出し完了: $EXPORT_HOST"
info "Jetson へのコピー例:"
echo "    scp $BUNDLE <user>@<jetson-ip>:~/tao-rtdetr-pipeline/jetson/models/"
echo "    （Jetson 側で）cd ~/tao-rtdetr-pipeline/jetson/models && tar -xzf jetson_bundle_${EXPERIMENT}.tar.gz"
