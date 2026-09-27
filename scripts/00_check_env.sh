#!/usr/bin/env bash
# 訓練サーバーの環境が TAO の要件を満たしているか確認する
#   - NVIDIA ドライバー（TAO 7.x は 595.45.04 以上、6.26.x は 580 以上）
#   - NVIDIA Container Toolkit 1.19.0 以上
#   - Docker 24 以上
#   - GPU のメモリ 16GB 以上（推奨 24GB）
source "$(dirname "$0")/common.sh"

failed=0

version_ge() {  # version_ge 現在 必要  → 現在 >= 必要 なら真
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

# 使うコンテナのタグから、必要なドライバーを決める
case "$TAO_IMAGE" in
  *:7.*) REQ_DRIVER="595.45.04" ;;
  *:6.*) REQ_DRIVER="580.0" ;;
  *)     REQ_DRIVER="580.0" ;;
esac
info "使用するコンテナ: $TAO_IMAGE（必要なドライバー: $REQ_DRIVER 以上）"

# ---- ドライバー ----
if DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"; then
  if version_ge "$DRIVER" "$REQ_DRIVER"; then
    ok "ドライバー: $DRIVER"
  else
    error "ドライバー $DRIVER は古すぎます（$REQ_DRIVER 以上が必要）→ ドライバーを更新してください"
    failed=1
  fi
else
  error "nvidia-smi が動きません"
  failed=1
fi

# ---- GPU ----
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null | \
while IFS=, read -r idx name mem cap; do
  mem_mib="${mem//[^0-9]/}"
  if (( mem_mib >= 16000 )); then
    ok "GPU$idx:$name /$mem / compute_cap$cap"
  else
    warn "GPU$idx:$name /$mem（16GB 未満。バッチサイズを下げてください）"
  fi
done

# ---- Container Toolkit ----
if CTK="$(nvidia-ctk --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)"; then
  if version_ge "$CTK" "1.19.0"; then
    ok "NVIDIA Container Toolkit: $CTK"
  else
    error "NVIDIA Container Toolkit $CTK は古すぎます（1.19.0 以上が必要）"
    failed=1
  fi
else
  error "nvidia-ctk が見つかりません"
  failed=1
fi

# ---- Docker ----
if DOCKER_VER="$(docker version --format '{{.Server.Version}}' 2>/dev/null)"; then
  if version_ge "$DOCKER_VER" "24.0.0"; then
    ok "Docker: $DOCKER_VER"
  else
    error "Docker $DOCKER_VER は古すぎます（24 以上が必要）"
    failed=1
  fi
else
  error "Docker に接続できません（sudo なしで使えるか: 'sudo usermod -aG docker \$USER' → 再ログイン）"
  failed=1
fi

# ---- コンテナから GPU が見えるか ----
if docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
  ok "コンテナから GPU を認識できます"
else
  error "コンテナから GPU が見えません（Container Toolkit の設定を確認）"
  failed=1
fi

# ---- ディスク容量（TAO コンテナだけで 20GB 以上）----
avail_gb="$(df -BG --output=avail "$REPO_DIR" | tail -n1 | tr -dc '0-9')"
if (( avail_gb >= 100 )); then
  ok "空きディスク: ${avail_gb}GB"
else
  warn "空きディスク: ${avail_gb}GB（コンテナ・データ・結果で 100GB 以上あると安心）"
fi

echo
if (( failed == 0 )); then
  ok "環境チェック完了。次は scripts/01_setup_ngc.sh"
else
  error "要件を満たしていない項目があります"
  exit 1
fi
