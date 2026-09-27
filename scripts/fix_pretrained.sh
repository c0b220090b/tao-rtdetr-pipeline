#!/usr/bin/env bash
# 事前学習済みチェックポイントのキー名を今の TAO に合わせて付け替える（TAO コンテナ内の torch を使う）
#
# 使い方:
#   ./scripts/fix_pretrained.sh models/trafficcamnet_transformer_lite_vtrainable_resnet50_v2.0/resnet50_trafficcamnet_rtdetr.pth
#
# 出力: 同じフォルダに <元の名前>_fixed.pth
# そのあと .env の PRETRAINED_MODEL に /workspace/... の形で出力ファイルを指定して訓練する。
source "$(dirname "$0")/common.sh"

SRC="${1:-}"
[[ -n "$SRC" && -f "$SRC" ]] || { error "入力の .pth を指定してください（リポジトリ内のパス）"; exit 1; }
SRC_ABS="$(readlink -f "$SRC")"
[[ "$SRC_ABS" == "$REPO_DIR"/* ]] || { error "リポジトリ内のファイルを指定してください（コンテナから見えないため）"; exit 1; }

DST_ABS="${SRC_ABS%.pth}_fixed.pth"
SRC_WS="${SRC_ABS/#$REPO_DIR/$WS}"
DST_WS="${DST_ABS/#$REPO_DIR/$WS}"

docker run --rm -v "$REPO_DIR:$WS" -w "$WS" --entrypoint python "$TAO_IMAGE" \
  "$WS/tools/fix_pretrained_keys.py" "$SRC_WS" "$DST_WS"

# 所有者を実行ユーザーに戻す
docker run --rm -v "$(dirname "$DST_ABS"):/d" --entrypoint chown "$TAO_IMAGE" \
  "$(id -u):$(id -g)" "/d/$(basename "$DST_ABS")" >/dev/null 2>&1 || true

echo
ok ".env に次を設定してください:"
echo "    PRETRAINED_MODEL=$DST_WS"
echo "    PRETRAINED_BACKBONE="
