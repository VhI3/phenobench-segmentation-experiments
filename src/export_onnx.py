import argparse
from pathlib import Path

import torch

from .data_phenobench import get_task_config
from .models import build_deeplab_mnv3


class SegmentationExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def parse_args():
    ap = argparse.ArgumentParser(description="Export a segmentation model to ONNX")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/model.onnx")
    ap.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "weed_binary"],
    )
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    return ap.parse_args()


def main():
    args = parse_args()
    task_config = get_task_config(args.task)

    model = build_deeplab_mnv3(num_classes=task_config.num_classes)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    export_model = SegmentationExportWrapper(model).eval()

    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy,
        str(out_path),
        opset_version=args.opset,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

    print(f"Exported ONNX model to: {out_path}")


if __name__ == "__main__":
    main()
