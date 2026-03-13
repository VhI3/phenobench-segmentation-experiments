import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_phenobench import PhenoBenchSemantics, get_task_config
from .models import build_deeplab_mnv3
from .metrics import IoUMeter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation on PhenoBench"
    )
    parser.add_argument("--data_root", default="./PhenoBench", type=str)
    parser.add_argument("--out_dir", default="./runs", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--task",
        default="multiclass",
        choices=["multiclass", "weed_binary"],
        type=str,
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    return parser.parse_args()


def save_ckpt(
    path: str, model: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int
):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, path)


def make_loaders(args):
    train_ds = PhenoBenchSemantics(
        args.data_root,
        split="train",
        image_size=args.image_size,
        augment=True,
        task=args.task,
    )
    val_ds = PhenoBenchSemantics(
        args.data_root,
        split="val",
        image_size=args.image_size,
        augment=False,
        task=args.task,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl


@torch.no_grad()
def run_validation(model, val_dl, device, num_classes):
    model.eval()
    meter = IoUMeter(num_classes=num_classes, ignore_index=-1)
    for img, mask in tqdm(val_dl, desc="val", leave=False):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)["out"][0]
        pred = logits.argmax(dim=0)
        meter.update(pred.cpu(), mask[0].cpu())
    iou = meter.iou_per_class().tolist()
    return iou, meter.miou()


def main():
    args = parse_args()
    device = torch.device(args.device)
    task_config = get_task_config(args.task)
    train_dl, val_dl = make_loaders(args)

    model = build_deeplab_mnv3(num_classes=task_config.num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"train epoch {epoch}/{args.epochs}")
        total_loss = 0.0

        for img, mask in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(img)["out"]
                loss = loss_fn(out, mask)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(train_dl))

        iou, miou = run_validation(model, val_dl, device, task_config.num_classes)

        print(
            f"[epoch {epoch}] task={args.task} train_loss={avg_loss:.4f} "
            f"val_mIoU={miou:.4f} classes={list(task_config.class_names)} IoU={iou}"
        )

        save_ckpt(str(out_dir / "ckpt_last.pt"), model, optim, epoch)
        if miou >= best_miou:
            best_miou = miou
            save_ckpt(str(out_dir / "ckpt_best.pt"), model, optim, epoch)

    print("Done. Checkpoints saved in:", out_dir)


if __name__ == "__main__":
    main()
