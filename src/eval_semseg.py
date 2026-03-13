import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_phenobench import PhenoBenchSemantics, get_task_config
from .models import build_deeplab_mnv3
from .metrics import IoUMeter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./PhenoBench")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "weed_binary"],
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    task_config = get_task_config(args.task)

    ds = PhenoBenchSemantics(
        args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        task=args.task,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = build_deeplab_mnv3(num_classes=task_config.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    meter = IoUMeter(num_classes=task_config.num_classes, ignore_index=-1)

    with torch.no_grad():
        for img, mask in tqdm(dl, desc=f"eval {args.split}"):
            img = img.to(device)
            logits = model(img)["out"][0]
            pred = logits.argmax(dim=0).cpu()
            meter.update(pred, mask[0].cpu())

    iou = meter.iou_per_class()
    miou = meter.miou()

    print("Classes:", list(task_config.class_names))
    print("IoU per class:", [round(x, 4) for x in iou.tolist()])
    print("mIoU:", round(miou, 4))


if __name__ == "__main__":
    main()
