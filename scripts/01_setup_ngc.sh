#!/usr/bin/env bash
# NGC（nvcr.io）にログインして、TAO のコンテナをダウンロードする
source "$(dirname "$0")/common.sh"

if [[ -z "${NGC_KEY:-}" ]]; then
  error ".env の NGC_KEY が空です。https://ngc.nvidia.com で API キーを発行して設定してください"
  exit 1
fi

info "nvcr.io にログインします"
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

info "TAO コンテナを取得します（20GB 以上あるので時間がかかります）: $TAO_IMAGE"
docker pull "$TAO_IMAGE"

info "コンテナ内に rtdetr コマンドがあるか確認します"
if docker run --rm --entrypoint which "$TAO_IMAGE" rtdetr; then
  ok "rtdetr コマンドを確認しました"
else
  error "コンテナに rtdetr コマンドが見つかりません（TAO_IMAGE のタグを確認してください）"
  exit 1
fi

ok "準備完了。次は scripts/02_download_pretrained.sh"
