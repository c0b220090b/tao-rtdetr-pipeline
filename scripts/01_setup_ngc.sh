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

info "コンテナ内で rtdetr コマンドが使えるか確認します"
docker run --rm "$TAO_IMAGE" rtdetr --help | head -n 20

ok "準備完了。次は scripts/02_download_pretrained.sh"
