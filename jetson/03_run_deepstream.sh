#!/usr/bin/env bash
# DeepStream で推論を実行する（Jetson 上で実行）
#
# 使い方:
#   ./03_run_deepstream.sh video.mp4                 # 動画ファイル → 画面に表示
#   ./03_run_deepstream.sh /dev/video0               # USB カメラ
#   ./03_run_deepstream.sh csi                       # CSI カメラ（IMX219 など）
#   ./03_run_deepstream.sh video.mp4 --sink file     # 結果を build/output.mp4 に保存
#   ./03_run_deepstream.sh video.mp4 --sink fake     # 表示しない（純粋な速度測定用）
#   オプション: --threshold 0.5  --fp32  --width 640 --height 640
#
# 画面に 5 秒ごとに FPS（**PERF: ...）が表示される。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"
MODELS="$HERE/models"
mkdir -p "$BUILD"

INPUT="${1:-}"
[[ -n "$INPUT" ]] || { sed -n 2,13p "$0"; exit 1; }
shift

SINK=display
THRESHOLD=0.5
PREC=fp16
W=640
H=640
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sink) SINK="$2"; shift ;;
    --threshold) THRESHOLD="$2"; shift ;;
    --fp32) PREC=fp32 ;;
    --width) W="$2"; shift ;;
    --height) H="$2"; shift ;;
    *) echo "不明なオプション: $1"; exit 1 ;;
  esac
  shift
done

ONNX="$MODELS/model.onnx"
LABELS="$MODELS/labels.txt"
PARSER_LIB="$BUILD/libnvds_infercustomparser_tao.so"
for f in "$ONNX" "$LABELS"; do
  [[ -f "$f" ]] || { echo "[ERROR] $f がありません（サーバーの export 結果を models/ に展開してください）"; exit 1; }
done
[[ -f "$PARSER_LIB" ]] || { echo "[ERROR] 先に ./01_build_parser.sh を実行してください"; exit 1; }

NUM_CLASSES="$(grep -c . "$LABELS")"
NETWORK_MODE=2; [[ "$PREC" == fp32 ]] && NETWORK_MODE=0
ENGINE="${ONNX}_b1_gpu0_${PREC}.engine"
[[ -f "$ENGINE" ]] || echo "[INFO] エンジンが無いので、初回起動時に DeepStream が作ります（数分かかります）"

# ---------- nvinfer の設定を生成 ----------
INFER_CONFIG="$BUILD/config_infer_primary_rtdetr.txt"
sed -e "s#@ONNX@#$ONNX#" -e "s#@ENGINE@#$ENGINE#" -e "s#@LABELS@#$LABELS#" \
    -e "s#@WIDTH@#$W#" -e "s#@HEIGHT@#$H#" -e "s#@NETWORK_MODE@#$NETWORK_MODE#" \
    -e "s#@NUM_CLASSES@#$NUM_CLASSES#" -e "s#@PARSER_LIB@#$PARSER_LIB#" \
    -e "s#@THRESHOLD@#$THRESHOLD#" \
    "$HERE/config_infer_primary_rtdetr.txt.in" > "$INFER_CONFIG"

# ---------- 入力（source）----------
LIVE=1
MUX_W=1280; MUX_H=720
if [[ "$INPUT" == csi ]]; then
  SOURCE="[source0]
enable=1
type=5
camera-csi-sensor-id=0
camera-width=1280
camera-height=720
camera-fps-n=30
camera-fps-d=1"
elif [[ "$INPUT" == /dev/video* ]]; then
  SOURCE="[source0]
enable=1
type=1
camera-v4l2-dev-node=${INPUT#/dev/video}
camera-width=1280
camera-height=720
camera-fps-n=30
camera-fps-d=1"
else
  [[ -f "$INPUT" ]] || { echo "[ERROR] 入力ファイルがありません: $INPUT"; exit 1; }
  LIVE=0
  MUX_W=1920; MUX_H=1080
  SOURCE="[source0]
enable=1
type=3
uri=file://$(readlink -f "$INPUT")
num-sources=1
gpu-id=0
cudadec-memtype=0"
fi

# ---------- 出力（sink）----------
case "$SINK" in
  display)
    [[ -n "${DISPLAY:-}" ]] || echo "[WARN] DISPLAY が未設定です（SSH なら --sink file か fake を使う）"
    SINK_SECTION="[sink0]
enable=1
type=2
sync=0
gpu-id=0"
    ;;
  file)
    # Orin Nano にはハードウェアエンコーダーが無いので、ソフトウェアエンコード（enc-type=1）
    SINK_SECTION="[sink0]
enable=1
type=3
container=1
codec=1
enc-type=1
sync=0
bitrate=4000000
output-file=$BUILD/output.mp4"
    ;;
  fake)
    SINK_SECTION="[sink0]
enable=1
type=1
sync=0"
    ;;
  *) echo "[ERROR] --sink は display / file / fake のどれか"; exit 1 ;;
esac

APP_CONFIG="$BUILD/deepstream_app_config.txt"
tmpl="$(<"$HERE/deepstream_app_config.txt.in")"
tmpl="${tmpl//@SOURCE_SECTION@/$SOURCE}"
tmpl="${tmpl//@SINK_SECTION@/$SINK_SECTION}"
tmpl="${tmpl//@LIVE@/$LIVE}"
tmpl="${tmpl//@MUX_WIDTH@/$MUX_W}"
tmpl="${tmpl//@MUX_HEIGHT@/$MUX_H}"
tmpl="${tmpl//@INFER_CONFIG@/$INFER_CONFIG}"
printf '%s\n' "$tmpl" > "$APP_CONFIG"

echo "[INFO] 設定: $APP_CONFIG"
echo "[INFO] クラス数（background 含む）: $NUM_CLASSES / 精度: $PREC / しきい値: $THRESHOLD"
deepstream-app -c "$APP_CONFIG"
if [[ "$SINK" == file ]]; then echo "[OK] 保存先: $BUILD/output.mp4"; fi
