"""
キーポイント検出モデルの定義。

商用利用を見据え、models/detector.py(SSDlite)と同じ方針で、torchvision公式の
BSD-3ライセンス重みのみを使用する。バックボーンはResNet50-FPN(ImageNet事前
学習)、検出ヘッド(分類・回帰・キーポイント)は完全スクラッチ学習。

num_keypointsはモデル全体で共有の1つの値であり、クラスごとに変えられない
(torchvisionのKeypointRCNNPredictorは全クラス共通の単一ヘッドのため)。
そのため、タスク内の全クラス中の最大点数(例: stadiumタスクではhome_plateの5)
にパディングして使う(パディング方法はutils/keypoint_dataset.pyを参照)。
"""

import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.resnet import ResNet50_Weights


def build_model(num_classes: int, num_keypoints: int, pretrained_backbone: bool = True):
    """num_classes には背景クラスを含める (= len(task.classes) + 1)。
    num_keypoints はモデル全体で共有の1つの値(= max(task.point_counts.values()))。
    """
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    model = keypointrcnn_resnet50_fpn(
        weights=None,                        # 検出ヘッド・キーポイントヘッドはスクラッチ
        weights_backbone=weights_backbone,   # バックボーンのみImageNet事前学習
        num_classes=num_classes,
        num_keypoints=num_keypoints,
    )
    return model


def load_checkpoint(weights_path: str, num_classes: int, num_keypoints: int, device: str = "cpu"):
    # weights_backboneの有無でbackboneのBatchNorm実装(FrozenBatchNorm2d vs
    # 通常のBatchNorm2d)が変わりstate_dictのキー集合が異なる(models/detector.pyの
    # reduced_tail問題とは別原因だが同じ対策が有効)。学習時と同じ構造にするため
    # 常にpretrained_backbone=Trueで構築してから、fine-tune済みのstate_dictで
    # 全重みを上書きする。
    model = build_model(num_classes, num_keypoints, pretrained_backbone=True)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
