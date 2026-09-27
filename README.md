# TAO RT-DETR パイプライン（GPU サーバーで訓練 → Jetson Orin Nano で推論）

NVIDIA TAO Toolkit で物体検出モデル **RT-DETR（ResNet50）** を自分のデータで訓練し、
Jetson Orin Nano の **DeepStream** で高速に推論するまでの手順とスクリプト一式です。

```
[GPU サーバー: RTX 3090 ×2]                           [Jetson Orin Nano]
 データ準備 → 訓練 → 評価 → ONNX 書き出し  ──scp──▶  TensorRT 変換 → DeepStream で推論
 (TAO コンテナ nvcr.io/nvidia/tao/tao-toolkit)          (JetPack 6.x + DeepStream 7.1)
```

- TAO はすべて **Docker コンテナの中**で動くので、サーバーに Python 環境を作る必要はありません。
- 設定は `.env`（よく変える値）と `specs/rtdetr_resnet50.yaml`（細かい設定）の 2 か所だけです。
- 実行した設定は `results/<実験名>/run_history.txt` に自動で記録されます。

---

## 0. 必要な環境

### 訓練サーバー

| 項目 | 要件 | 備考 |
|---|---|---|
| GPU | Volta 以降・VRAM 16GB 以上（推奨 24GB） | RTX 3090 ×2 で OK |
| NVIDIA ドライバー | **TAO 7.x は 595.45.04 以上** / TAO 6.26.x は 580 以上 | 550 では動かない |
| NVIDIA Container Toolkit | TAO 7.x は 1.19.0 以上 | 6.26.x は古めでも動く |
| Docker | 24 以上 | |
| ディスク | 100GB 以上の空き | コンテナだけで 20GB 以上 |
| NGC アカウント | API キー | https://ngc.nvidia.com で無料登録 |

ドライバーと Container Toolkit の更新は `tools/update_nvidia.sh` でできます（`sudo ./tools/update_nvidia.sh --dry-run` で内容を確認してから実行）。
ドライバーを 595 にできない場合は、`.env` の `TAO_IMAGE` を `6.26.3-pyt` にしてください。

### Jetson

- JetPack 6.x（6.2 推奨）＋ DeepStream 7.1
- `sudo nvpmodel -m 2 && sudo jetson_clocks`（Orin Nano の MAXN SUPER モード）

---

## 1. 初期設定（サーバー、最初の 1 回だけ）

```bash
git clone <このリポジトリ> tao-rtdetr-pipeline
cd tao-rtdetr-pipeline
cp .env.example .env
vim .env                          # NGC_KEY を設定。GPU_IDS=0,1 など

./scripts/00_check_env.sh         # ドライバー・Docker・GPU の確認
./scripts/01_setup_ngc.sh         # nvcr.io にログインし、TAO コンテナを取得
```

## 2. 事前学習済みモデルを取得する

ゼロから学習すると大量のデータと時間が必要になるので、NVIDIA が学習済みのモデルから始めます（転移学習）。
NGC CLI（`ngc` コマンド）が必要です。無ければスクリプトがインストール手順を表示します。

```bash
./scripts/02_download_pretrained.sh --list     # RT-DETR 系のモデルを探す
./scripts/02_download_pretrained.sh nvidia/trafficcamnet_transformer_lite:trainable_resnet50_v2.0
```

表示された `/workspace/models/.../*.pth` のパスを `.env` の `PRETRAINED_MODEL=` に書きます。
**事前学習モデルの backbone と spec の `model.backbone`（既定は `resnet_50`）を合わせてください。**

## 3. データを準備する

COCO 形式か YOLO 形式のデータを `data/raw/` に置いて、TAO 用に変換します。

```bash
# COCO 形式（1 つの JSON を train/val に 9:1 で自動分割）
python3 scripts/03_prepare_dataset.py coco --json data/raw/anno.json --images data/raw/images

# COCO 形式（train / val が分かれている）
python3 scripts/03_prepare_dataset.py coco \
  --json data/raw/train.json --images data/raw/train \
  --val-json data/raw/val.json --val-images data/raw/val

# YOLO 形式（images/ と labels/。train/val のサブフォルダがあればそのまま使う）
python3 scripts/03_prepare_dataset.py yolo --root data/raw/yolo --names data/raw/yolo/data.yaml
```

必要なもの: `pip install pillow`（YOLO の data.yaml を使う場合は `pyyaml` も）

出力（`data/processed/`）:

| ファイル | 内容 |
|---|---|
| `images/train/`, `images/val/` | 画像（ハードリンクなので容量はほぼ増えない） |
| `annotations/train.json`, `val.json` | COCO 形式。クラス ID は **1〜N に振り直し** |
| `classmap.txt` | クラス名（1 行 1 クラス） |
| `num_classes.txt` | `dataset.num_classes` に使う値 = **クラス数 + 1** |

> **なぜ +1 なのか**：TAO の RT-DETR はカスタムデータでは `category_id` をそのままクラス番号として使うため、
> `num_classes` は「最大の ID + 1」が必要です。ID を 1〜N にしているので N+1 になり、0 番は未使用の
> `background` になります（書き出し時の `labels.txt` も 1 行目が background）。スクリプトが自動で処理します。

変換後、クラスごとのボックス数が表示されます。極端に少ないクラスがないか確認してください。

## 4. 訓練する

```bash
tmux new -s tao                     # SSH が切れても止まらないように
./scripts/10_train.sh
```

- 結果は `results/<EXPERIMENT>/train/` に保存されます（チェックポイント `.pth` とログ）。
- `.env` の `GPU_IDS=0,1` で 2 枚の GPU を使って並列に学習します（DDP）。
- spec の値はコマンドラインで追加上書きできます：`./scripts/10_train.sh train.optim.lr=5e-5`
- 途中で止まったら再開できます：`./scripts/10_train.sh --resume`
- 別の条件で試すときは `.env` の `EXPERIMENT` を変えれば、結果が別フォルダに分かれます。

GPU の様子は別のターミナルで `watch -n 1 nvidia-smi` で確認できます。

### 転移学習の方法を選ぶ

| やり方 | コマンド | 向いているとき |
|---|---|---|
| 全体を学習（既定） | `./scripts/10_train.sh` | データが数千枚以上ある |
| クラス判定の層だけ作り直して全体を学習 | `./scripts/fix_pretrained.sh --reinit-class-head <.pth>` で作った `_fixed_newhead.pth` を `PRETRAINED_MODEL` に | 事前学習とクラスが違う（ほとんどの場合） |
| クラス判定の層だけ学習（他は固定） | `./scripts/10_train.sh "$(./scripts/freeze_presets.sh head_only)"` | データが少ない・事前学習と似た物体 |
| 枠の層も含めて頭だけ学習 | `./scripts/10_train.sh "$(./scripts/freeze_presets.sh heads)"` | 上より少し柔軟にしたい |

固定した部品は `results/<実験名>/train/status.json` に `Freezed module [...]` と記録されます。
2 段階（頭だけ数エポック → できたチェックポイントを `PRETRAINED_MODEL` にして全体を学習）にすると、安定しやすくなります。

## 5. 評価・目視確認

```bash
./scripts/20_evaluate.sh            # val の mAP（最新のチェックポイント）
./scripts/30_inference.sh           # val 画像にボックスを描いて保存 → results/<EXPERIMENT>/inference/
./scripts/20_evaluate.sh /workspace/results/rtdetr_r50_v1/train/model_epoch_050.pth   # 特定のエポック
```

## 6. ONNX に書き出す

```bash
./scripts/40_export.sh
```

`results/<EXPERIMENT>/export/` に `model.onnx`・`labels.txt`・`nvdsinfer_config.yaml` ができ、
Jetson に持っていく `jetson_bundle_<EXPERIMENT>.tar.gz` にまとめられます。

## 7. Jetson で推論する

Jetson にもこのリポジトリを clone して、書き出したファイルを `jetson/models/` に展開します。

```bash
# サーバー側
scp results/rtdetr_r50_v1/jetson_bundle_rtdetr_r50_v1.tar.gz <user>@<jetson-ip>:~/tao-rtdetr-pipeline/jetson/models/

# Jetson 側
cd ~/tao-rtdetr-pipeline/jetson
(cd models && tar -xzf jetson_bundle_*.tar.gz)

sudo nvpmodel -m 2 && sudo jetson_clocks     # 性能モードを最大に

./01_build_parser.sh        # DETR 用の後処理ライブラリをビルド（最初の 1 回）
./02_build_engine.sh        # ONNX → TensorRT(FP16) に変換し、速度を表示
./03_run_deepstream.sh sample.mp4                  # 動画ファイル → 画面表示
./03_run_deepstream.sh /dev/video0                 # USB カメラ
./03_run_deepstream.sh csi                         # CSI カメラ
./03_run_deepstream.sh sample.mp4 --sink file      # jetson/build/output.mp4 に保存
./03_run_deepstream.sh sample.mp4 --sink fake      # 表示なし（速度測定用）
```

5 秒ごとに `**PERF:` として FPS が表示されます。

---

## ディレクトリ構成

```
.
├── .env.example              # 設定のひな形（cp して .env を作る）
├── specs/rtdetr_resnet50.yaml  # TAO の実験設定
├── scripts/
│   ├── common.sh             # 共通処理（TAO コンテナの起動など）
│   ├── 00_check_env.sh       # 環境チェック
│   ├── 01_setup_ngc.sh       # NGC ログイン・コンテナ取得
│   ├── 02_download_pretrained.sh
│   ├── 03_prepare_dataset.py # COCO / YOLO → TAO 用 COCO
│   ├── 10_train.sh
│   ├── 20_evaluate.sh
│   ├── 30_inference.sh
│   └── 40_export.sh
├── tools/update_nvidia.sh     # ドライバー・Container Toolkit の更新
├── jetson/
│   ├── 01_build_parser.sh    # DeepStream 用後処理ライブラリ
│   ├── 02_build_engine.sh    # trtexec で TensorRT エンジン化
│   ├── 03_run_deepstream.sh  # DeepStream 実行
│   ├── *.txt.in              # 設定テンプレート
│   └── models/               # ここに export 結果を置く（git 管理外）
├── data/raw/                 # 元データ（git 管理外）
├── data/processed/           # 変換後データ（git 管理外）
├── models/                   # 事前学習モデル（git 管理外）
└── results/                  # 訓練結果（git 管理外）
```

データ・モデル・結果・`.env` は `.gitignore` に入っているので、コミットされるのは手順とスクリプトだけです。

---

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| `CUDA driver version is insufficient` | ドライバーが古い。595 以上に更新するか、`TAO_IMAGE` を `6.26.3-pyt` に |
| `unauthorized` でコンテナが取れない | `NGC_KEY` を確認して `./scripts/01_setup_ngc.sh` をやり直す |
| `CUDA out of memory` | `.env` の `BATCH_SIZE` を下げる（4 → 2） |
| DataLoader が `shared memory` でエラー | コンテナは `--ipc=host` で起動済み。`dataset.workers=4` に下げる |
| 訓練後に「バックボーンが読み込まれていません」と出る / 1 エポック目の val mAP がほぼ 0 | 事前学習の重みとキー名がずれている。`./scripts/fix_pretrained.sh <.pth>` で付け替えた `_fixed.pth` を `PRETRAINED_MODEL` に指定 |
| 精度が上がらない | 事前学習モデルを使っているか確認。データ量（1 クラス数百ボックス以上が目安）とエポック数を増やす |
| `num_classes` の不一致エラー | データを入れ替えたら `03_prepare_dataset.py` をやり直す（`num_classes.txt` が更新される） |
| Jetson で FP16 だと検出が出ない / NaN | `./02_build_engine.sh --fp32` と `./03_run_deepstream.sh ... --fp32` で FP32 にする。ResNet 以外の backbone（ConvNeXt 系）は FP16 で不安定になりやすい |
| Jetson でボックスがずれる | `export` と Jetson の入力サイズ（`--width/--height`）を揃える |
| Jetson で遅い | 入力サイズを下げる（例: 480x480 で訓練し直す）、`nvpmodel -m 2` を確認、`jtop` で温度を確認 |

## TAO 7 のエージェント方式について

TAO 7.x からは、Claude Code などのコーディングエージェントに自然文で指示して訓練する方式が推奨になりました
（`/plugin marketplace add NVIDIA-TAO/tao-skill-bank@7.2.0`）。このリポジトリはその裏側と同じ
「TAO コンテナを直接実行する」方式をスクリプト化したものです。手順を固定して再現できるのが利点です。

## 参考

- [TAO Toolkit ドキュメント](https://docs.nvidia.com/tao/tao-toolkit/latest/index.html)
- [RT-DETR（TAO）](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/object_detection/rt_detr.html)
- [TAO リリースノート（コンテナのタグ・ドライバー要件）](https://docs.nvidia.com/tao/tao-toolkit/latest/text/release_notes.html)
- [deepstream_tao_apps（後処理ライブラリ）](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps)
