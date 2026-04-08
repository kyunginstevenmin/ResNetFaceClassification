"""
model.py — Model definitions for the MLP Head experiment.

Import this in train.py, inference scripts, or notebooks:
    from model import ResNet18, MLPHead, HEAD_CONFIGS, freeze_backbone, load_backbone_only, topk_accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    """Basic residual block used in ResNet18."""
    def __init__(self, num_channels, use_1x1conv=False, kernel_sizes=3, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_sizes, stride=strides,
                                   padding=(kernel_sizes - 1) // 2)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_sizes, stride=1,
                                   padding=(kernel_sizes - 1) // 2)
        self.conv3 = nn.LazyConv2d(num_channels, 1, stride=strides) if use_1x1conv else None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

HEAD_CONFIGS = {
    'A': {'hidden_dims': [],           'dropout': 0.4},  # baseline: 512 -> 7000
    'B': {'hidden_dims': [1024],       'dropout': 0.4},  # wider:    512 -> 1024 -> 7000
    'C': {'hidden_dims': [1024, 512],  'dropout': 0.4},  # two-layer:512 -> 1024 -> 512 -> 7000
}


class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_dims, num_classes, dropout=0.4):
        super().__init__()
        layers = []
        current_dim = in_features
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(current_dim, hidden_dim, bias=False),  # bias=False: BN subsumes it
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))  # no activation — raw logits
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    def __init__(self, arch, lr=0.1, num_classes=7001, head=None, num_features=512):
        super().__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(b[0], b[1], first_block=(i == 0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ))
        if head is None:
            head = MLPHead(num_features, hidden_dims=[], num_classes=num_classes)
        self.head = head

    def forward(self, x, return_feats=False):
        # return_feats kept for API compatibility — always returns raw logits
        return self.head(self.net(x))


class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=7001, head=None):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                         head=head, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model):
    """Freeze all parameters except model.head. Backbone stays frozen for the
    entire run — it was trained on the same dataset and task."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)


def load_backbone_only(checkpoint_path, model, device):
    """Load backbone weights from an existing baseline checkpoint.

    The baseline uses the OLD architecture where the classifier lives inside
    net.last.2 (LazyLinear). That key is excluded here so the new MLPHead is
    randomly initialised. Checkpoint key is 'model_state_dict'.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_state = checkpoint['model_state_dict']
    backbone_state = {k: v for k, v in saved_state.items()
                      if not k.startswith('net.last.2')}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if unexpected:
        print(f"Warning: ignoring {len(unexpected)} unexpected keys: {unexpected[:5]}")
    print(f"Backbone loaded. New head randomly initialized ({len(missing)} keys.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def topk_accuracy(logits, targets, topk=(1, 5)):
    """Return top-k accuracy (%) for each k, as a list of scalar tensors."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()                           # (maxk, N)
    correct = pred.eq(targets.unsqueeze(0))   # (maxk, N)
    results = []
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().sum()
        results.append(correct_k * (100.0 / targets.size(0)))
    return results
