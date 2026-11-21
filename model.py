import timm
import torch.nn as nn


def build_model(name="efficientnet_b0", pretrained=True, drop_rate=0.3) -> nn.Module:
    """Single-logit binary classifier: sigmoid(output) = P(fake)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)
