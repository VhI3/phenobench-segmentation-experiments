import argparse
import time

import torch

from .data_phenobench import get_task_config
from .models import build_deeplab_mnv3


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark segmentation runtime")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "weed_binary"],
    )
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    task_config = get_task_config(args.task)

    model = build_deeplab_mnv3(num_classes=task_config.num_classes).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    x = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / args.iters
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print("task:", args.task)
    print("device:", device)
    print("image_size:", args.image_size)
    print("iterations:", args.iters)
    print("avg_latency_ms:", round(avg_ms, 2))
    print("fps:", round(fps, 2))


if __name__ == "__main__":
    main()
