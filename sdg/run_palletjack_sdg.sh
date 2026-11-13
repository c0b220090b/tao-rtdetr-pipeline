#!/usr/bin/env bash
# Isaac Sim（コンテナ）で、ハンドパレット（pallet jack）の合成データを作る
#
# 使い方:
#   ./sdg/run_palletjack_sdg.sh 30          # お試し（30 枚 × 3 種類）。まずこれで動くか確認
#   ./sdg/run_palletjack_sdg.sh 2000        # 本番（2000 + 2000 + 1000 = 5000 枚）
#   GPU=1 ./sdg/run_palletjack_sdg.sh 2000  # 2 枚目の GPU を使う（訓練と同時に動かすとき）
#
# 生成スクリプトは NVIDIA 公式のもの（NVIDIA-AI-IOT/synthetic_data_generation_training_workflow）をそのまま使う。
# この公式スクリプトは Isaac Sim 4.2.0 向けに書かれているので、コンテナも 4.2.0 に合わせている。
#
# 出力: data/raw/palletjack_sdg/{distractors_warehouse,distractors_additional,no_distractors}/
#       （KITTI 形式。scripts/kitti_to_coco.py で COCO 形式に変換する）
#
# 注意:
#   - 3D モデルはインターネット（NVIDIA の S3）から読み込むので、ネット接続が必要
#   - 初回はシェーダーのコンパイルで 10〜20 分ほど何も進まないように見えることがある（キャッシュされるので 2 回目から速い）
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:-30}"
GPU="${GPU:-0}"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:4.2.0}"
SRC="$REPO_DIR/data/raw/sdg_src"
OUT="$REPO_DIR/data/raw/palletjack_sdg"
CACHE="$HOME/.cache/isaac-sim-docker"

if [[ ! -d "$SRC" ]]; then
  git clone --depth 1 https://github.com/NVIDIA-AI-IOT/synthetic_data_generation_training_workflow.git "$SRC"
fi
mkdir -p "$OUT" "$CACHE"/{kit,ov,pip,glcache,computecache,logs,data,documents}

run() {  # run <サブフォルダ名> <枚数> <distractors>
  local name="$1" frames="$2" dist="$3"
  echo "[INFO] 生成: $name（$frames 枚, distractors=$dist）"
  docker run --rm --gpus "\"device=$GPU\"" --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v "$CACHE/kit:/isaac-sim/kit/cache:rw" \
    -v "$CACHE/ov:/root/.cache/ov:rw" \
    -v "$CACHE/pip:/root/.cache/pip:rw" \
    -v "$CACHE/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "$CACHE/computecache:/root/.nv/ComputeCache:rw" \
    -v "$CACHE/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "$CACHE/data:/root/.local/share/ov/data:rw" \
    -v "$CACHE/documents:/root/Documents:rw" \
    -v "$SRC:/sdg_src:ro" -v "$OUT:/out:rw" \
    --entrypoint /isaac-sim/python.sh "$IMAGE" \
    /sdg_src/palletjack_sdg/standalone_palletjack_sdg.py \
    --headless True --height 544 --width 960 \
    --num_frames "$frames" --distractors "$dist" --data_dir "/out/$name"
}

# 公式の generate_data.sh と同じ 2 : 2 : 1 の割合
run distractors_warehouse  "$N"           warehouse
run distractors_additional "$N"           additional
run no_distractors         "$(( N / 2 ))" None

# コンテナは root で書き込むので、所有者を自分に戻す
docker run --rm -v "$OUT:/out" --entrypoint chown "$IMAGE" -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true

echo "[OK] 出力: $OUT"
find "$OUT" -maxdepth 3 -type d | sort
echo "画像の枚数: $(find "$OUT" -name '*.png' | wc -l)"
