#!/usr/bin/env bash
# RT-DETR を訓練する
#
# 使い方:
#   ./scripts/10_train.sh                     # .env の設定で訓練
#   ./scripts/10_train.sh --resume            # 最後のチェックポイントから再開
#   ./scripts/10_train.sh train.optim.lr=5e-5 # spec の値を追加で上書き（何個でも）
#
# 結果: results/<EXPERIMENT>/train/ にチェックポイント(.pth)とログ
# 長時間かかるので、SSH 切断に備えて tmux / screen の中で実行するのがおすすめ。
source "$(dirname "$0")/common.sh"

extra=()
resume=0
for a in "$@"; do
  if [[ "$a" == "--resume" ]]; then resume=1; else extra+=("$a"); fi
done

args=(
  "train.num_gpus=$NUM_GPUS"
  "train.gpu_ids=[$(seq -s, 0 $((NUM_GPUS - 1)))]"   # コンテナ内では 0 から振り直される
  "train.num_epochs=$NUM_EPOCHS"
  "dataset.batch_size=$BATCH_SIZE"
)

if [[ -n "$PRETRAINED_MODEL" && -n "$PRETRAINED_BACKBONE" ]]; then
  warn "PRETRAINED_MODEL と PRETRAINED_BACKBONE の両方が指定されています。TAO は PRETRAINED_BACKBONE を無視します"
fi
if [[ -n "$PRETRAINED_MODEL" ]]; then
  args+=("train.pretrained_model_path=$PRETRAINED_MODEL")
  info "事前学習モデル（検出器全体）から開始: $PRETRAINED_MODEL"
elif [[ -n "$PRETRAINED_BACKBONE" ]]; then
  args+=("model.pretrained_backbone_path=$PRETRAINED_BACKBONE")
  info "事前学習バックボーンから開始: $PRETRAINED_BACKBONE"
else
  warn "事前学習の重みが指定されていません。ゼロから学習するので精度が出にくくなります"
fi

if (( resume )); then
  ckpt="$(find_checkpoint)"
  args+=("train.resume_training_checkpoint_path=$ckpt")
  info "再開: $ckpt"
fi

mkdir -p "$RESULTS_HOST"
# 実行した設定を記録しておく（再現用）
{
  echo "date: $(date -Iseconds)"
  echo "image: $TAO_IMAGE"
  echo "git: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo none)"
  echo "args: $(common_overrides) ${args[*]} ${extra[*]:-}"
} >> "$RESULTS_HOST/run_history.txt"

TRAIN_LOG="$RESULTS_HOST/train_console_$(date +%Y%m%d-%H%M%S).log"
info "画面出力の保存先: $TRAIN_LOG"
# shellcheck disable=SC2046
TAO_LOG="$TRAIN_LOG" tao_run rtdetr train -e "$SPEC" $(common_overrides) "${args[@]}" "${extra[@]}"

# 事前学習の重みがちゃんと読み込まれたか確認する
STATUS="$RESULTS_HOST/train/status.json"
if [[ -f "$STATUS" ]] && grep -q "missing_keys=\['model.backbone" "$STATUS"; then
  warn "事前学習の重みのうち、バックボーンが読み込まれていません（status.json の missing_keys を参照）"
  warn "→ ./scripts/fix_pretrained.sh <事前学習の.pth> でキー名を付け替え、できた _fixed.pth を PRETRAINED_MODEL にして再訓練してください"
fi

ok "訓練完了: $RESULTS_HOST/train"
info "次は ./scripts/20_evaluate.sh"
