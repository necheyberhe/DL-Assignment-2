import torch
import torch.nn as nn
import torch.nn.functional as F


class KochBackbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            #experiment1
            #  nn.Sigmoid(),
            #experiment2
            nn.ReLU(inplace=True),
            nn.Linear(1024, embedding_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        z = self.fc(x)
        return F.normalize(z, p=2, dim=1)


class SiameseBCEHead(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, z1, z2):
        l1 = torch.abs(z1 - z2)
        logits = self.classifier(l1).squeeze(1)
        return logits
# backbone = KochBackbone(embedding_dim=128)
# head = SiameseBCEHead(embedding_dim=128)

# x1 = torch.randn(4, 3, 105, 105)
# x2 = torch.randn(4, 3, 105, 105)

# z1 = backbone(x1)
# z2 = backbone(x2)
# logits = head(z1, z2)

# print("z1:", z1.shape)
# print("z2:", z2.shape)
# print("logits:", logits.shape)