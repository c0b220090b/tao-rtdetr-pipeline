#!/usr/bin/env python3
"""
事前学習済みの RT-DETR チェックポイントの「重みの名前（キー）」を、今の TAO が期待する形に揃える。

背景:
  TrafficCamNet Transformer Lite（trainable_resnet50_v2.0）の resnet50_trafficcamnet_rtdetr.pth は、
  バックボーンのキーが "model.model.backbone.conv1.weight" のように "model." が 2 重になっている。
  TAO 6.26.3 は "model.backbone.conv1.weight" を探すため一致せず、ResNet50 部分が読み込まれない
  （status.json に missing_keys=['model.backbone....'] と出る）。

やること:
  エンコーダー（".encoder."）のキーの前置き（例: "model."）を基準にして、
  バックボーンのキーの前置きを同じ形に付け替える。その他のキーはそのまま。

使い方（torch が必要なので TAO コンテナの中で実行する。scripts/fix_pretrained.sh 経由が簡単）:
  python3 tools/fix_pretrained_keys.py <入力.pth> <出力.pth>
"""
from __future__ import annotations

import sys
from collections import Counter


def fix_keys(keys: list[str]) -> dict[str, str]:
    """{元のキー: 新しいキー} を返す（変わらないキーも含む）。"""
    enc = next((k for k in keys if ".encoder." in k or k.startswith("encoder.")), None)
    if enc is None:
        raise SystemExit("エンコーダーのキー（.encoder.）が見つかりません。RT-DETR のチェックポイントか確認してください")
    ref_prefix = enc[: enc.index("encoder.")]          # 例: "model."

    mapping = {}
    for k in keys:
        if "backbone." in k:
            new = ref_prefix + k[k.index("backbone."):]  # 例: "model.model.backbone.x" → "model.backbone.x"
        else:
            new = k
        mapping[k] = new
    dup = [k for k, n in Counter(mapping.values()).items() if n > 1]
    if dup:
        raise SystemExit(f"付け替え後にキーが重複します（例: {dup[:3]}）。手動で確認してください")
    return mapping


def prefix_summary(keys) -> Counter:
    """キーの先頭 2 階層ごとの件数（確認用）"""
    return Counter(".".join(k.split(".")[:3]) for k in keys)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    has_sd = isinstance(ck, dict) and "state_dict" in ck
    sd = ck["state_dict"] if has_sd else ck

    keys = list(sd.keys())
    mapping = fix_keys(keys)
    changed = sum(1 for k, v in mapping.items() if k != v)

    print(f"キーの数: {len(keys)}  /  付け替え: {changed}")
    if changed == 0:
        enc = next(k for k in keys if "encoder." in k)
        bb = next((k for k in keys if "backbone." in k), "(なし)")
        sys.exit(f"付け替えるキーがありませんでした（エンコーダー例: {enc} / バックボーン例: {bb}）。"
                 "バックボーンとエンコーダーの前置きが同じなので、別の原因です")
    print("付け替えの例:")
    for k, v in list(mapping.items()):
        if k != v:
            print(f"  {k}\n    → {v}")
            break
    new_sd = type(sd)((mapping[k], v) for k, v in sd.items())

    print("\n付け替え後の構成（先頭 3 階層: 件数）:")
    for p, n in sorted(prefix_summary(new_sd.keys()).items()):
        print(f"  {p:<45s} {n}")

    if has_sd:
        ck = dict(ck)
        ck["state_dict"] = new_sd
    else:
        ck = new_sd
    torch.save(ck, dst)
    print(f"\n保存しました: {dst}")


if __name__ == "__main__":
    main()
