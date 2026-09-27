#!/usr/bin/env bash
# =============================================================================
# NVIDIA ドライバー & NVIDIA Container Toolkit 更新スクリプト（Ubuntu 22.04 向け）
#
# 使い方:
#   sudo ./update_nvidia.sh                  # 更新（最新ドライバーを自動選択）
#   sudo ./update_nvidia.sh --dry-run        # 実行せずに、やることだけ表示
#   sudo ./update_nvidia.sh --driver nvidia-driver-580   # ドライバーを指定
#   sudo ./update_nvidia.sh --yes            # 確認プロンプトを省略
#   sudo ./update_nvidia.sh --verify         # 再起動後の動作確認だけ実行
#
# 流れ:
#   0. 事前チェック（GPU使用中のプロセスやコンテナがないか）と現状のログ保存
#   1. NVIDIA Container Toolkit を更新
#   2. NVIDIA ドライバーを更新
#   3. 再起動 → 再起動後に --verify で確認
# =============================================================================
set -euo pipefail

DRY_RUN=0
ASSUME_YES=0
VERIFY_ONLY=0
TARGET_DRIVER=""

# 動作確認に使うコンテナ（古いCUDAと新しいCUDAの両方が動くかを見る）
TEST_IMAGES=(
  "nvidia/cuda:12.2.2-base-ubuntu22.04"
  "nvidia/cuda:12.8.1-base-ubuntu22.04"
)

# ---------- 共通関数 ----------
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  (dry-run) $*"
  else
    eval "$*"
  fi
}

confirm() {
  [[ $ASSUME_YES -eq 1 ]] && return 0
  read -r -p "$1 [y/N]: " ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

# ---------- 引数の処理 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y)  ASSUME_YES=1 ;;
    --verify)  VERIFY_ONLY=1 ;;
    --driver)  TARGET_DRIVER="${2:-}"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) error "不明なオプション: $1"; exit 1 ;;
  esac
  shift
done

if [[ $EUID -ne 0 ]]; then
  error "sudo を付けて実行してください: sudo $0 $*"
  exit 1
fi

# =============================================================================
# 再起動後の動作確認
# =============================================================================
verify() {
  info "===== 動作確認 ====="
  local failed=0

  if nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi 正常"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
  else
    error "nvidia-smi が失敗しました（ドライバーが読み込まれていません）"
    failed=1
  fi

  if command -v nvidia-ctk >/dev/null 2>&1; then
    ok "NVIDIA Container Toolkit: $(nvidia-ctk --version | head -n1)"
  else
    error "nvidia-ctk が見つかりません"
    failed=1
  fi

  for img in "${TEST_IMAGES[@]}"; do
    info "コンテナからGPUが見えるか確認: $img"
    if docker run --rm --gpus all "$img" nvidia-smi -L; then
      ok "$img で GPU を認識"
    else
      error "$img でGPUを認識できませんでした"
      failed=1
    fi
  done

  if [[ $failed -eq 0 ]]; then
    ok "すべての確認に成功しました。普段使っているコンテナも一度起動して確認してください。"
  else
    error "失敗した項目があります。上のメッセージを確認してください。"
    exit 1
  fi
}

if [[ $VERIFY_ONLY -eq 1 ]]; then
  verify
  exit 0
fi

# =============================================================================
# 0. 事前チェックとログ保存
# =============================================================================
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  error "Ubuntu 以外は対象外です（検出: ${PRETTY_NAME:-不明}）"
  exit 1
fi
info "OS: $PRETTY_NAME / カーネル: $(uname -r)"

REAL_USER="${SUDO_USER:-root}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
LOG_DIR="$REAL_HOME/nvidia-update-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
info "更新前の状態を保存します: $LOG_DIR"
{
  nvidia-smi || true
  echo; nvidia-ctk --version || true
} > "$LOG_DIR/before_nvidia.txt" 2>&1
dpkg -l | grep -Ei "nvidia|cuda" > "$LOG_DIR/before_packages.txt" 2>/dev/null || true
docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$LOG_DIR/before_containers.txt" 2>&1 || true
chown -R "$REAL_USER": "$LOG_DIR"

CURRENT_DRIVER="$(dpkg-query -W -f='${Package}\n' 'nvidia-driver-*' 2>/dev/null \
  | grep -E '^nvidia-driver-[0-9]+(-open)?$' | head -n1 || true)"
info "現在のドライバーパッケージ: ${CURRENT_DRIVER:-不明}"

# GPU を使っているプロセス（計算ジョブ）がないか
GPU_PROCS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
if [[ -n "$GPU_PROCS" ]]; then
  warn "GPU を使っている計算プロセスがあります。ドライバー更新で停止します:"
  echo "$GPU_PROCS"
  confirm "このまま続行しますか？" || { info "中止しました"; exit 0; }
fi

# 動いているコンテナがないか
RUNNING="$(docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null || true)"
if [[ -n "$RUNNING" ]]; then
  warn "動作中のコンテナがあります。Docker 再起動・再起動時に停止します:"
  echo "$RUNNING"
  confirm "このまま続行しますか？" || { info "中止しました"; exit 0; }
fi

# =============================================================================
# 1. NVIDIA Container Toolkit の更新
# =============================================================================
info "===== 1. NVIDIA Container Toolkit ====="
REPO_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

# 公式リポジトリが未設定、または古い形式なら設定し直す
if [[ ! -f "$REPO_LIST" ]] || ! grep -q "nvidia.github.io/libnvidia-container/stable" "$REPO_LIST"; then
  info "NVIDIA の公式リポジトリを設定します"
  run "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o $KEYRING"
  run "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=$KEYRING] https://#g' > $REPO_LIST"
fi

run "apt-get update"
info "更新予定のバージョン:"
apt-cache policy nvidia-container-toolkit | sed -n '1,3p'
run "apt-get install -y nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1"
run "nvidia-ctk runtime configure --runtime=docker"

# =============================================================================
# 2. NVIDIA ドライバーの更新
# =============================================================================
info "===== 2. NVIDIA ドライバー ====="
if [[ -z "$TARGET_DRIVER" ]]; then
  # 今と同じ系統（通常版 or -open版）の中で一番新しいものを選ぶ
  SUFFIX=""
  [[ "$CURRENT_DRIVER" == *-open ]] && SUFFIX="-open"
  TARGET_DRIVER="$(apt-cache search --names-only "^nvidia-driver-[0-9]+${SUFFIX}\$" \
    | awk '{print $1}' | grep -E "^nvidia-driver-[0-9]+${SUFFIX}\$" \
    | sort -t- -k3 -n | tail -n1)"
fi

if [[ -z "$TARGET_DRIVER" ]]; then
  error "インストール可能なドライバーが見つかりませんでした"
  exit 1
fi

info "インストール可能なドライバー（参考）:"
ubuntu-drivers list 2>/dev/null | grep -E "^nvidia-driver" | sort -V || true
info "更新先: $TARGET_DRIVER （現在: ${CURRENT_DRIVER:-不明}）"

if [[ "$TARGET_DRIVER" == "$CURRENT_DRIVER" ]]; then
  ok "ドライバーはすでに最新系統です。パッケージの更新のみ行います。"
fi

confirm "$TARGET_DRIVER をインストールしますか？" || { info "ドライバー更新をスキップしました"; exit 0; }

# DKMS でカーネルモジュールをビルドするためにヘッダーを入れておく
run "apt-get install -y linux-headers-$(uname -r)"
run "apt-get install -y $TARGET_DRIVER"

# =============================================================================
# 3. 再起動
# =============================================================================
info "===== 3. 再起動 ====="
info "更新前の状態は $LOG_DIR に保存しました。"
info "再起動後に次のコマンドで動作確認してください:"
echo "    sudo $(realpath "$0") --verify"

if [[ $DRY_RUN -eq 1 ]]; then
  info "dry-run のため再起動しません"
  exit 0
fi

if confirm "今すぐ再起動しますか？"; then
  reboot
else
  warn "後で必ず再起動してください（再起動するまで新しいドライバーは使われません）"
fi
