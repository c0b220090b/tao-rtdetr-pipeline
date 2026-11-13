#!/usr/bin/env bash
# ONNX → TensorRT エンジンに変換し、速度を測る（Jetson 上で実行）
#
# 使い方:
#   ./02_build_engine.sh                    # FP16（おすすめ）
#   ./02_build_engine.sh --fp32             # FP16 で検出が出ない・NaN になるとき
#   ./02_build_engine.sh --width 960 --height 544
#
# エンジンは「実際に推論する環境」で作る必要がある（サーバーで作ったものは Jetson で使えない）。
# ファイル名は DeepStream が自動で作る名前と同じにしてあるので、そのまま config から読まれる。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$HERE/models"
ONNX="$MODELS/model.onnx"
PREC=fp16
W=640
H=640

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp32) PREC=fp32 ;;
    --fp16) PREC=fp16 ;;
    --width) W="$2"; shift ;;
    --height) H="$2"; shift ;;
    --onnx) ONNX="$2"; shift ;;
    *) echo "不明なオプション: $1"; exit 1 ;;
  esac
  shift
done

[[ -f "$ONNX" ]] || { echo "[ERROR] $ONNX がありません（サーバーの export 結果をコピーしてください）"; exit 1; }
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[[ -x "$TRTEXEC" ]] || { echo "[ERROR] trtexec がありません（sudo apt install tensorrt）"; exit 1; }

# DeepStream の命名規則: <onnxファイル名>_b<batch>_gpu<id>_<precision>.engine
ENGINE="${ONNX}_b1_gpu0_${PREC}.engine"

prec_flag=()
[[ "$PREC" == fp16 ]] && prec_flag=(--fp16)

echo "[INFO] 変換: $ONNX → $ENGINE（$PREC, 入力 1x3x${H}x${W}）"
echo "[INFO] Orin Nano では 5〜15 分ほどかかります"
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  "${prec_flag[@]}" \
  --minShapes=inputs:1x3x${H}x${W} \
  --optShapes=inputs:1x3x${H}x${W} \
  --maxShapes=inputs:1x3x${H}x${W} \
  --memPoolSize=workspace:2048 \
  2>&1 | tee "$MODELS/trtexec_build_${PREC}.log" | grep -E "Throughput|Latency: min|GPU Compute Time: min|\[E\]|PASSED|FAILED" || true

[[ -f "$ENGINE" ]] || { echo "[ERROR] エンジンの作成に失敗しました。ログ: $MODELS/trtexec_build_${PREC}.log"; exit 1; }
echo "[OK] $ENGINE"
echo "[INFO] 上の Throughput（qps）が、モデル単体の 1 秒あたりの推論回数です"
