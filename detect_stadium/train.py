"""
アノテーション済みラベル(bbox_label_web.pyで作成)を使って物体検出モデルを学習する。

使い方:
    python train.py --task stadium
    python train.py --task player --epochs 80 --batch-size 8

学習結果は runs/train/<task>/<name>/ に best.pt / last.pt / classes.json として保存される。
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.detector import build_model
from utils.detection_dataset import (
    DetectionDataset,
    build_splits,
    collate_fn,
    get_eval_transforms,
    get_train_transforms,
)
from utils.label_store import load_tasks


def move_targets(targets, device):
    return [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]


def train_one_epoch(model, loader, optimizer, device, epoch, epochs, writer, global_step):
    model.train()
    running = {}
    pbar = tqdm(loader, ncols=140, desc=f"Epoch {epoch}/{epochs} [train]")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = move_targets(targets, device)

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v.item()
        pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items()})

        if writer is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)
        global_step += 1

    n = len(loader)
    return {k: v / max(1, n) for k, v in running.items()}, global_step


@torch.inference_mode()
def validate(model, loader, device):
    """SSD/FCOSはeval()時にlossを返さないため、train()状態のままlossだけ計算する。
    ただしBatchNormはバッチ統計を使うとバッチサイズ1で落ちるため、running statsに固定する。"""
    if loader is None:
        return None
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    running = {}
    for images, targets in tqdm(loader, ncols=140, desc="          [val]"):
        images = [img.to(device) for img in images]
        targets = move_targets(targets, device)
        loss_dict = model(images, targets)
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v.item()
    n = len(loader)
    losses = {k: v / max(1, n) for k, v in running.items()}
    losses["total"] = sum(losses.values())
    return losses


def parse_args():
    p = argparse.ArgumentParser(description="矩形検出モデルの学習")
    p.add_argument("--task", required=True, choices=["stadium", "player"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--videos-dir", default=None, help="省略時はclasses.yamlの値")
    p.add_argument("--labels-dir", default=None, help="省略時はclasses.yamlの値")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3,
                    help="学習率をlrまで線形にウォームアップするエポック数。0で無効")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default="exp", help="実行名 (runs/train/<task>/<name>)")
    p.add_argument("--project", default=str(ROOT / "runs" / "train"))
    p.add_argument("--resume", type=str, default="", help="続きから学習する重みファイルパス")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=15,
                    help="val lossがこのエポック数改善しなければ打ち切る。0で無効")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    task = tasks[args.task]
    if args.videos_dir:
        task.videos_dir = args.videos_dir
    if args.labels_dir:
        task.labels_dir = args.labels_dir
    num_classes = len(task.classes) + 1  # +背景

    train_samples, val_samples = build_splits(task, val_ratio=args.val_ratio)
    print(f"train samples: {len(train_samples)}  val samples: {len(val_samples)}")
    if len(train_samples) == 0:
        print("学習データがありません。bbox_label_web.py でアノテーションしてください。")
        return

    train_ds = DetectionDataset(train_samples, transforms=get_train_transforms(task.classes))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, drop_last=len(train_ds) > args.batch_size,
    )

    val_loader = None
    if val_samples:
        val_ds = DetectionDataset(val_samples, transforms=get_eval_transforms())
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn,
        )

    device = args.device
    # weights_backboneの有無でバックボーンのreduced_tailが変わり構造が
    # 変わってしまうため、resume時も含め常にpretrained_backbone=Trueで構築する
    # (resume時はこの直後にstate_dictで上書きされる)。
    model = build_model(num_classes, pretrained_backbone=True).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resume from {args.resume}")

    # BatchNorm/biasにはweight decayをかけない(過学習抑制と精度の両立に効きやすい)
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.project) / args.task / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(task.classes, f, ensure_ascii=False, indent=2)

    writer = SummaryWriter(str(save_dir / "logs"))
    global_step = 0
    best_val = float("inf")
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_losses, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, writer, global_step,
        )
        print(f"[epoch {epoch}] train: " + " ".join(f"{k}={v:.4f}" for k, v in train_losses.items()))

        val_losses = validate(model, val_loader, device)
        if val_losses is not None:
            print(f"[epoch {epoch}] val:   " + " ".join(f"{k}={v:.4f}" for k, v in val_losses.items()))
            for k, v in val_losses.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), save_dir / "last.pt")
        metric = val_losses["total"] if val_losses is not None else sum(train_losses.values())
        improved = metric < best_val
        if improved:
            best_val = metric
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"  -> best.pt を更新 (metric={metric:.4f})")

        if val_losses is None:
            improved = True  # valが無い場合はearly stoppingを無効化する
        epochs_without_improve = 0 if improved else epochs_without_improve + 1
        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(f"[epoch {epoch}] val lossが{args.patience}エポック改善しなかったため学習を打ち切ります。")
            break

    print(f"学習完了。重みは {save_dir} に保存されました。")


if __name__ == "__main__":
    main()
