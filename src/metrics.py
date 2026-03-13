from dataclasses import dataclass
import torch


@dataclass
class IoUMeter:
    num_classes: int
    ignore_index: int = -1

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.conf = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.view(-1).to(torch.int64)
        target = target.view(-1).to(torch.int64)

        if self.ignore_index >= 0:
            valid = target != self.ignore_index
            pred = pred[valid]
            target = target[valid]

        in_range = (target >= 0) & (target < self.num_classes)
        pred = pred[in_range]
        target = target[in_range]

        idx = target * self.num_classes + pred
        hist = torch.bincount(
            idx,
            minlength=self.num_classes * self.num_classes,
        )
        self.conf += hist.reshape(self.num_classes, self.num_classes).cpu()

    def iou_per_class(self):
        conf = self.conf.to(torch.float64)

        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        denom = tp + fp + fn

        out = torch.zeros_like(denom)
        valid = denom > 0
        out[valid] = tp[valid] / denom[valid]
        return out

    def miou(self):
        iou = self.iou_per_class()

        gt_pixels = self.conf.sum(dim=1)
        seen = gt_pixels > 0
        if torch.any(seen):
            return iou[seen].mean().item()
        return 0.0
