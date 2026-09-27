#!/usr/bin/env bash
# train.freeze（固定する部品）のプリセット。10_train.sh の引数にそのまま渡す。
#
# 使い方:
#   ./scripts/10_train.sh "$(./scripts/freeze_presets.sh head_only)"
#
# プリセット:
#   head_only       クラス判定の層（dec/enc_score_head, denoising_class_embed）だけ学習。他は全部固定
#   heads           クラス判定＋枠の位置の層だけ学習（バックボーン・エンコーダー・デコーダー本体を固定）
#   backbone        バックボーンだけ固定（エンコーダー・デコーダーは学習）
#   none            何も固定しない（既定と同じ）
set -euo pipefail
case "${1:-}" in
  head_only) echo "train.freeze=[backbone,encoder,decoder.input_proj,decoder.decoder,decoder.query_pos_head,decoder.enc_output,decoder.enc_bbox_head,decoder.dec_bbox_head]" ;;
  heads)     echo "train.freeze=[backbone,encoder,decoder.input_proj,decoder.decoder,decoder.query_pos_head,decoder.enc_output]" ;;
  backbone)  echo "train.freeze=[backbone]" ;;
  none)      echo "train.freeze=[]" ;;
  *) sed -n 2,12p "$0" >&2; exit 1 ;;
esac
