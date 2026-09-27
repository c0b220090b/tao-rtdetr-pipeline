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

if [[ -n "$PRETRAINED_MODEL" ]]; then
  args+=("train.pretrained_model_path=$PRETRAINED_MODEL")
  info "事前学習モデルから開始: $PRETRAINED_MODEL"
else
  warn "PRETRAINED_MODEL が空です。ゼロから学習するので精度が出にくくなります"
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

# shellcheck disable=SC2046
tao_run rtdetr train -e "$SPEC" $(common_overrides) "${args[@]}" "${extra[@]}"

ok "訓練完了: $RESULTS_HOST/train"
info "次は ./scripts/20_evaluate.sh"
