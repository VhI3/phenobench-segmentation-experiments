import torch
import torchvision


def build_deeplab_mnv3(num_classes: int = 3) -> torch.nn.Module:
    """
    Build a lightweight DeepLabV3 model with a MobileNetV3 backbone.

    We start from torchvision's implementation and only swap the final
    classifier layer so the output channel count matches our dataset classes.
    This keeps the model setup simple and easy to tweak.
    """
    # No pretrained weights by default; training starts from scratch.
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)

    # Replace the final 1x1 conv in the segmentation head:
    # original out-channels -> `num_classes`.
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model
