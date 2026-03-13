import argparse
import time
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .data_phenobench import get_task_config
from .models import build_deeplab_mnv3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--out_png", default="pred_overlay.png", type=str)
    parser.add_argument("--tile", default=1024, type=int)
    parser.add_argument("--overlap", default=128, type=int)
    parser.add_argument("--emulate_5mp", action="store_true")
    parser.add_argument(
        "--task",
        default="multiclass",
        choices=["multiclass", "weed_binary"],
        type=str,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    return parser.parse_args()


def normalize(img_t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def tile_inference(model, img_t, tile=1024, overlap=128, device="cuda"):
    model.eval().to(device)
    _, H, W = img_t.shape
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile must be bigger than overlap")

    first_h = min(tile, H)
    first_w = min(tile, W)
    sample = model(img_t[:, :first_h, :first_w].unsqueeze(0).to(device))["out"]
    num_classes = sample.shape[1]

    logits_sum = torch.zeros((num_classes, H, W), device=device)
    votes = torch.zeros((1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile, H)
            x2 = min(x + tile, W)

            crop = img_t[:, y:y2, x:x2]
            pad_h = tile - (y2 - y)
            pad_w = tile - (x2 - x)

            h, w = crop.shape[-2], crop.shape[-1]
            pad_mode = "reflect"
            if pad_h >= h or pad_w >= w:
                pad_mode = "replicate"
            crop = F.pad(crop, (0, pad_w, 0, pad_h), mode=pad_mode)

            logits = model(crop.unsqueeze(0).to(device))["out"][0]
            logits = logits[:, :(y2 - y), :(x2 - x)]

            logits_sum[:, y:y2, x:x2] += logits
            votes[:, y:y2, x:x2] += 1

    return logits_sum / votes


def overlay_mask(rgb: np.ndarray, pred: np.ndarray, task: str) -> np.ndarray:
    if task == "weed_binary":
        colors = {
            0: (80, 60, 40),
            1: (0, 0, 255),
        }
    else:
        colors = {
            0: (80, 60, 40),
            1: (0, 200, 0),
            2: (0, 0, 255),
        }
    mask_color = np.zeros_like(rgb)
    for cls, color in colors.items():
        mask_color[pred == cls] = color
    return cv2.addWeighted(rgb, 0.65, mask_color, 0.35, 0.0)


def main():
    args = parse_args()
    device = torch.device(args.device)
    task_config = get_task_config(args.task)

    model = build_deeplab_mnv3(num_classes=task_config.num_classes).to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)[:, :, ::-1].copy()
    img_t = TF.to_tensor(img)

    if args.emulate_5mp:
        img_t = F.interpolate(
            img_t.unsqueeze(0),
            size=(1944, 2592),
            mode="bilinear",
            align_corners=False,
        )[0]
        img_np = cv2.resize(img_np, (2592, 1944), interpolation=cv2.INTER_LINEAR)

    img_t = normalize(img_t)

    t0 = time.perf_counter()
    logits = tile_inference(model, img_t, tile=args.tile, overlap=args.overlap, device=str(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    pred = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    out = overlay_mask(img_np, pred, task=args.task)

    cv2.imwrite(args.out_png, out)
    print(f"Saved: {args.out_png}")
    print(f"Tiled inference time: {(t1 - t0) * 1000:.1f} ms  | image shape: {pred.shape}")


if __name__ == "__main__":
    main()
