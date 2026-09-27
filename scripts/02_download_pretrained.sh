#!/usr/bin/env bash
# NGC から事前学習済みモデルをダウンロードして models/ に置く
#
# 使い方:
#   ./scripts/02_download_pretrained.sh --list                 # RT-DETR 系のモデルを検索
#   ./scripts/02_download_pretrained.sh <組織/モデル:バージョン>
#
# 例（交通カメラ向けに NVIDIA が学習済みの RT-DETR / ResNet50。人・車などの検出に強い）:
#   ./scripts/02_download_pretrained.sh nvidia/trafficcamnet_transformer_lite:trainable_resnet50_v2.0
#   （モデル名やバージョンは更新されることがあるので、先に --list で確認する）
#
# ダウンロード後、表示された .pth のパスを .env の PRETRAINED_MODEL に書く。
# ※ 事前学習モデルと同じ backbone（resnet_50）を spec で使うこと。
source "$(dirname "$0")/common.sh"

# NGC CLI（ngc コマンド）はホストにあればそれを使い、なければ公式の手順でインストールしてもらう
if ! command -v ngc >/dev/null 2>&1; then
  error "ngc コマンドがありません。次の手順でインストールしてください:"
  cat <<'EOF'
    wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/latest/files/ngccli_linux.zip -O ngccli_linux.zip
    unzip ngccli_linux.zip
    chmod u+x ngc-cli/ngc
    echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bashrc && source ~/.bashrc
    ngc config set     # API キーを入力（org は nvidia を選択）
EOF
  exit 1
fi

if [[ "${1:-}" == "--list" || -z "${1:-}" ]]; then
  info "RT-DETR 系の事前学習モデル一覧:"
  ngc registry model list "nvidia/tao/*rtdetr*" --format_type ascii || true
  ngc registry model list "nvidia/tao/*transformer*" --format_type ascii || true
  ngc registry model list "nvidia/*transformer_lite*" --format_type ascii || true
  echo
  info "使いたいモデルが見つかったら、バージョン一覧を確認:"
  echo "    ngc registry model list nvidia/tao/<モデル名>:*"
  exit 0
fi

TARGET="$1"
DEST="$REPO_DIR/models"
mkdir -p "$DEST"
info "ダウンロード: $TARGET → $DEST"
ngc registry model download-version "$TARGET" --dest "$DEST"

echo
info "ダウンロードされた重みファイル:"
find "$DEST" -name '*.pth' | sort | while read -r f; do
  echo "    ${f/#$REPO_DIR/$WS}"
done
echo
ok "上のパス（/workspace/... の形）を .env の PRETRAINED_MODEL に設定してください"
