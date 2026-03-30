from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import ResNet18_Weights, resnet18


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


@dataclass
class ModelConfig:
    pretrained: bool = True
    feature_dim: int = 512
    num_domains: int = 2
    domain_loss_weight: float = 0.2


class DeceptionDANN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(cfg.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(cfg.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, cfg.num_domains),
        )

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        return h

    def forward(self, x: torch.Tensor, lambd: float = 0.0):
        feat = self.extract(x)
        class_logits = self.classifier(feat)
        domain_logits = self.domain_head(grad_reverse(feat, lambd))
        return class_logits, domain_logits
