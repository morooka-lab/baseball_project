"""
物体検出モデルの定義。

商用利用を見据え、検出ヘッド(分類・回帰)は完全にスクラッチ学習し、
バックボーン(MobileNetV3)のみ ImageNet 事前学習重み(BSD-3ライセンス,
torchvision公式配布)を再利用する。モデル自体は torchvision の
SSDlite320-MobileNetV3-Large（軽量・高速）。
"""

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights


def build_model(num_classes: int, pretrained_backbone: bool = True):
    """num_classes には背景クラスを含める (= len(task.classes) + 1)。"""
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    model = ssdlite320_mobilenet_v3_large(
        weights=None,                      # 検出ヘッドはスクラッチ
        weights_backbone=weights_backbone,  # バックボーンのみImageNet事前学習
        num_classes=num_classes,
    )
    return model


def load_checkpoint(weights_path: str, num_classes: int, device: str = "cpu"):
    # torchvisionの ssdlite320_mobilenet_v3_large は weights_backbone の有無で
    # バックボーンの reduced_tail (チャンネル数)が変わり、学習時と異なる構造に
    # なってしまう。学習時と同じ構造にするため pretrained_backbone=True で
    # 構築してから、fine-tune済みのstate_dictで全重みを上書きする。
    model = build_model(num_classes, pretrained_backbone=True)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
