#!/usr/bin/env bash
# DeepStream で TAO の RT-DETR を使うための「後処理ライブラリ」をビルドする（Jetson 上で実行）
#
# TAO の RT-DETR は ONNX の出力が pred_logits / pred_boxes の 2 つで、
# DeepStream 標準の後処理では読めない。NVIDIA 公式の deepstream_tao_apps に入っている
# NvDsInferParseCustomDDETRTAO（DETR 系共通のパーサー。TAO が export 時に生成する設定も
# これを指定している）をビルドして使う。
#
# 出力: jetson/build/libnvds_infercustomparser_tao.so
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"
mkdir -p "$BUILD"

command -v deepstream-app >/dev/null || { echo "[ERROR] DeepStream が入っていません"; exit 1; }

# DeepStream のバージョン → deepstream_tao_apps のブランチ
DS_VER="$(deepstream-app --version-all 2>/dev/null | awk '/DeepStreamSDK/ {print $2}' | cut -d. -f1,2)"
case "$DS_VER" in
  7.1) BRANCH=release/tao_ds7.1ga ;;
  7.0) BRANCH=release/tao5.3_ds7.0ga ;;
  8.0) BRANCH=release/tao_ds8.0ga ;;
  9.*) BRANCH=release/tao_ds9.0ga ;;
  *)   echo "[ERROR] 未対応の DeepStream バージョン: '$DS_VER'"; exit 1 ;;
esac

# CUDA のバージョン（JetPack 6.2 なら 12.6）
CUDA_VER="$(readlink -f /usr/local/cuda | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
[[ -n "$CUDA_VER" ]] || { echo "[ERROR] /usr/local/cuda が見つかりません"; exit 1; }

echo "[INFO] DeepStream $DS_VER / CUDA $CUDA_VER / ブランチ $BRANCH"

SRC="$BUILD/deepstream_tao_apps"
if [[ ! -d "$SRC" ]]; then
  git clone --depth 1 -b "$BRANCH" https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git "$SRC"
fi

make -C "$SRC/post_processor" CUDA_VER="$CUDA_VER"
cp "$SRC/post_processor/libnvds_infercustomparser_tao.so" "$BUILD/"

# パーサー関数が入っているか確認
if nm -D "$BUILD/libnvds_infercustomparser_tao.so" | grep -q NvDsInferParseCustomDDETRTAO; then
  echo "[OK] $BUILD/libnvds_infercustomparser_tao.so（NvDsInferParseCustomDDETRTAO を含む）"
else
  echo "[ERROR] ビルドしたライブラリにパーサー関数が見つかりません"; exit 1
fi
