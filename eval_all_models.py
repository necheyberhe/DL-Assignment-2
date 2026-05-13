import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms, models

from Data.datasets import PairDataset
from Models.models import KochBackbone, SiameseBCEHead
from Models.resnet import ResNet18Backbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

transform_105 = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_loader_105 = DataLoader(
    PairDataset(pairs_test, root_dir, transform_105),
    batch_size=32,
    shuffle=False
)

test_loader_224 = DataLoader(
    PairDataset(pairs_test, root_dir, transform_224),
    batch_size=32,
    shuffle=False
)


def clean_state_dict(state):
    return {
        k: v for k, v in state.items()
        if "total_ops" not in k and "total_params" not in k
    }


def load_backbone(model, path):
    ckpt = torch.load(path, map_location=device)
    state = clean_state_dict(ckpt["backbone_state_dict"])
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_bce(path):
    ckpt = torch.load(path, map_location=device)

    backbone = KochBackbone(128).to(device)
    head = SiameseBCEHead(128).to(device)

    backbone.load_state_dict(clean_state_dict(ckpt["backbone_state_dict"]), strict=False)
    head.load_state_dict(clean_state_dict(ckpt["head_state_dict"]), strict=False)

    backbone.eval()
    head.eval()

    return backbone, head


@torch.no_grad()
def get_distance_scores(model, loader):
    scores, labels = [], []

    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)

        z1 = model(x1)
        z2 = model(x2)

        d = F.pairwise_distance(z1, z2, p=2)
        sim = -d

        scores.append(sim.cpu())
        labels.append(y)

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def get_bce_scores(model_tuple, loader):
    backbone, head = model_tuple

    scores, labels = [], []

    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)

        z1 = backbone(x1)
        z2 = backbone(x2)

        logits = head(z1, z2)
        probs = torch.sigmoid(logits)

        scores.append(probs.cpu())
        labels.append(y)

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def get_pretrained_scores(model, loader):
    scores, labels = [], []

    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)

        z1 = model(x1)
        z2 = model(x2)

        sim = F.cosine_similarity(z1, z2)

        scores.append(sim.cpu())
        labels.append(y)

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


class FrozenResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*list(base.children())[:-1])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return F.normalize(x, p=2, dim=1)


models_list = [
    # Experiment 1
    (
        "BCE_L1",
        load_bce("checkpoints/best_bce_l1.pt"),
        test_loader_105,
        get_bce_scores,
    ),
    (
        "Contrastive",
        load_backbone(KochBackbone(128).to(device), "checkpoints/best_contrastive.pt"),
        test_loader_105,
        get_distance_scores,
    ),
    (
        "Triplet_Random",
        load_backbone(KochBackbone(128).to(device), "checkpoints/best_triplet_random_margin_0.2.pt"),
        test_loader_105,
        get_distance_scores,
    ),
    (
        "Triplet_Semihard",
        load_backbone(KochBackbone(128).to(device), "checkpoints/best_triplet_semihard_margin_0.2.pt"),
        test_loader_105,
        get_distance_scores,
    ),

    # Experiment 2
    (
        "KochCNN_Exp2",
        load_backbone(KochBackbone(128).to(device), "checkpoints/exp2_best_koch.pt"),
        test_loader_105,
        get_distance_scores,
    ),
    (
        "ResNet18_Scratch",
        load_backbone(ResNet18Backbone(128).to(device), "checkpoints/exp2_best_resnet18.pt"),
        test_loader_105,
        get_distance_scores,
    ),

    # Experiment 3
    (
        "ResNet18_Pretrained_Frozen",
        FrozenResNet18().to(device).eval(),
        test_loader_224,
        get_pretrained_scores,
    ),
]


plt.figure(figsize=(10, 7))
rows = []

for name, model, loader, scorer in models_list:
    print("Evaluating", name)

    scores, labels = scorer(model, loader)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    rows.append({"model": name, "auc": roc_auc})

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison Across All Models")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/all_models_roc.png", dpi=400)
plt.close()

df = pd.DataFrame(rows)
df.to_csv("outputs/all_models_auc.csv", index=False)

print("\nAUC results:")
print(df)