#!/usr/bin/env python3
"""
事前学習済みの RT-DETR チェックポイントを、TAO が正しく読み込める「キー名」に作り直す。

背景（TAO 6.26.3 で確認）:
  train.pretrained_model_path の重みは、TAO の rtdetr_parser でキー名が変換されてから読み込まれる。
  rtdetr_parser には次の規則がある（nvidia_tao_pytorch/cv/rtdetr/model/utils.py）:
      "module" を含むキー         → 先頭の 1 階層を削る
      "model.model.backbone." で始まる → その前置きを丸ごと削る（バックボーン単体のファイル向け）
      "model." で始まる            → "model." を 1 つ削る
  TrafficCamNet Transformer Lite の resnet50_trafficcamnet_rtdetr.pth はすべてのキーが
  "model.model." で始まるため、
      model.model.encoder.x        → model.encoder.x        （一致する）
      model.model.backbone.conv1.w → conv1.w                （一致しない → ResNet50 が読み込まれない）
  となる。

やること:
  TAO が探しているキー（model.backbone.* / model.encoder.* / model.decoder.*）の頭に
  "module." を付けて保存する。rtdetr_parser の最初の規則で "module." だけが削られ、
  すべてのキーが正しい名前になる。

オプション --reinit-class-head:
  クラス判定の層（dec_score_head / enc_score_head / denoising_class_embed）を保存しない。
  TAO はこれらを初期化し直して学習する（事前学習とクラスの意味が違うときに使う）。

使い方（torch が必要なので TAO コンテナの中で実行する。scripts/fix_pretrained.sh 経由が簡単）:
  python3 tools/fix_pretrained_keys.py [--reinit-class-head] <入力.pth> <出力.pth>
"""
from __future__ import annotations

import sys
from collections import Counter


def target_key(k: str) -> str:
    """元のキー → TAO が探しているキー（model.backbone.* など）"""
    while k.startswith("model.model."):
        k = k[len("model."):]
    if k.startswith("module."):
        k = k[len("module."):]
    return k


def fix_keys(keys: list[str]) -> dict[str, str]:
    """{元のキー: 保存するキー} を返す。保存するキーは rtdetr_parser を通ると target_key になる。"""
    mapping = {k: "module." + target_key(k) for k in keys}
    dup = [k for k, n in Counter(mapping.values()).items() if n > 1]
    if dup:
        raise SystemExit(f"作り直したキーが重複します（例: {dup[:3]}）。手動で確認してください")
    return mapping


CLASS_HEAD_PATTERNS = ("dec_score_head.", "enc_score_head.", "denoising_class_embed.")


def is_class_head(k: str) -> bool:
    return any(p in k for p in CLASS_HEAD_PATTERNS)


def prefix_summary(keys) -> Counter:
    """キーの先頭 2 階層ごとの件数（確認用）"""
    return Counter(".".join(k.split(".")[:2]) for k in keys)


def main():
    args = sys.argv[1:]
    reinit = "--reinit-class-head" in args
    args = [a for a in args if a != "--reinit-class-head"]
    if len(args) != 2:
        sys.exit(__doc__)
    src, dst = args

    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        sys.exit("想定外のファイル形式です")
    if "tao_model" in ck:
        print("[WARN] tao_model 付きのチェックポイントです。TAO は別の読み込み経路を使うので、この変換は不要かもしれません")
    sd = ck.get("state_dict", ck.get("model", ck))

    if reinit:
        dropped = [k for k in sd if is_class_head(k)]
        sd = type(sd)((k, v) for k, v in sd.items() if not is_class_head(k))
        print(f"クラス判定の層を {len(dropped)} 個取り除きました（TAO が初期化し直します）")
        if not dropped:
            sys.exit("クラス判定の層が見つかりませんでした")

    keys = list(sd.keys())
    mapping = fix_keys(keys)
    targets = [target_key(k) for k in keys]

    print(f"キーの数: {len(keys)}")
    print("変換の例:")
    shown = set()
    for k in keys:
        part = target_key(k).split(".")[1] if "." in target_key(k) else k
        if part in shown:
            continue
        shown.add(part)
        print(f"  {k}\n    → 保存: {mapping[k]}\n    → TAO 読み込み後: {target_key(k)}")
    print("\nTAO が読み込んだあとの構成（先頭 2 階層: 件数）:")
    for p, n in sorted(prefix_summary(targets).items()):
        print(f"  {p:<30s} {n}")

    new_sd = type(sd)((mapping[k], v) for k, v in sd.items())
    torch.save({"state_dict": new_sd}, dst)
    print(f"\n保存しました: {dst}")


if __name__ == "__main__":
    main()
