#!/usr/bin/env bash
# shellcheck disable=SC2034  # ここで定義した変数は source 先のスクリプトで使う
# 各スクリプトから source される共通処理
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }

# ---------- .env の読み込み ----------
if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
else
  warn ".env がありません。'cp .env.example .env' で作成してください（デフォルト値で続行）"
fi

TAO_IMAGE="${TAO_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt}"
EXPERIMENT="${EXPERIMENT:-rtdetr_r50_v1}"
GPU_IDS="${GPU_IDS:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-72}"
BATCH_SIZE="${BATCH_SIZE:-4}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
INPUT_HEIGHT="${INPUT_HEIGHT:-640}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-}"

# コンテナ内のパス（リポジトリ全体を /workspace にマウントする）
WS=/workspace
SPEC="$WS/specs/rtdetr_resnet50.yaml"
RESULTS_HOST="$REPO_DIR/results/$EXPERIMENT"
RESULTS_DIR="$WS/results/$EXPERIMENT"

NUM_GPUS="$(awk -F, '{print NF}' <<< "$GPU_IDS")"

# データ準備スクリプトが書き出したクラス数（背景を含む = クラス数 + 1）
num_classes() {
  local f="$REPO_DIR/data/processed/num_classes.txt"
  if [[ ! -f "$f" ]]; then
    error "$f がありません。先に scripts/03_prepare_dataset.py を実行してください"
    exit 1
  fi
  cat "$f"
}

# 全コマンド共通の上書き（データのパスは spec に固定で書いてある）
common_overrides() {
  echo "results_dir=$RESULTS_DIR" \
       "dataset.num_classes=$(num_classes)" \
       "dataset.augmentation.train_spatial_size=[$INPUT_HEIGHT,$INPUT_WIDTH]" \
       "dataset.augmentation.eval_spatial_size=[$INPUT_HEIGHT,$INPUT_WIDTH]"
}

# 最新（または指定）のチェックポイントを探す
find_checkpoint() {
  local ckpt="${1:-}"
  if [[ -n "$ckpt" ]]; then
    echo "${ckpt/#$REPO_DIR/$WS}"; return   # ホストのパスで渡されてもコンテナ内のパスに直す
  fi
  ckpt="$(find "$RESULTS_HOST/train" -name '*.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n1 | cut -d' ' -f2-)"
  if [[ -z "$ckpt" ]]; then
    error "チェックポイント(.pth)が $RESULTS_HOST/train に見つかりません"
    exit 1
  fi
  # ホストのパス → コンテナ内のパス
  echo "${ckpt/#$REPO_DIR/$WS}"
}

# ---------- TAO コンテナでコマンドを実行する ----------
# 使い方: tao_run rtdetr train -e spec.yaml key=value ...
tao_run() {
  mkdir -p "$REPO_DIR/.cache/home"
  local tty_flag=() rc=0
  [[ -t 1 ]] && tty_flag=(-it)

  info "実行: $*"
  # TAO_LOG が指定されていれば、画面出力をファイルにも保存する
  local log="${TAO_LOG:-/dev/null}"
  docker run --rm "${tty_flag[@]}" \
    --gpus "\"device=$GPU_IDS\"" \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO_DIR:$WS" \
    -w "$WS" \
    -e HOME="$WS/.cache/home" \
    "$TAO_IMAGE" \
    "$@" 2>&1 | tee -a "$log" || rc=$?

  # コンテナは root で動くので、生成物の所有者を実行ユーザーに戻す
  docker run --rm -v "$REPO_DIR/results:/r" -v "$REPO_DIR/.cache:/c" --entrypoint chown \
    "$TAO_IMAGE" -R "$(id -u):$(id -g)" /r /c >/dev/null 2>&1 || true
  return $rc
}
