import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        base = resnet18(weights=None)  # no ImageNet pretraining

        self.feature_extractor = nn.Sequential(
            *list(base.children())[:-1]
        )

        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return F.normalize(z, p=2, dim=1)