import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models

from Data.datasets import PairDataset
from Models.models import KochBackbone
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
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_loader_105 = DataLoader(
    PairDataset(pairs_test, root_dir, transform_105),
    batch_size=32, shuffle=False
)

test_loader_224 = DataLoader(
    PairDataset(pairs_test, root_dir, transform_224),
    batch_size=32, shuffle=False
)


def load_clean(model, path):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["backbone_state_dict"]

    state = {k:v for k,v in state.items()
             if "total_ops" not in k and "total_params" not in k}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def get_scores(model, loader):
    scores, labels = [], []

    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)

        z1 = model(x1)
        z2 = model(x2)

        d = F.pairwise_distance(z1, z2)
        sim = -d

        scores.append(sim.cpu())
        labels.append(y)

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def get_scores_pretrained(model, loader):
    scores, labels = [], []

    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)

        z1 = model(x1)
        z2 = model(x2)

        sim = F.cosine_similarity(z1, z2)

        scores.append(sim.cpu())
        labels.append(y)

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


# ---------- Models ----------

models_list = []

# Exp2 Koch
koch = load_clean(
    KochBackbone(128).to(device),
    "checkpoints/exp2_best_koch.pt"
)
models_list.append(("KochCNN", koch, test_loader_105, get_scores))

# Exp2 ResNet
resnet = load_clean(
    ResNet18Backbone(128).to(device),
    "checkpoints/exp2_best_resnet18.pt"
)
models_list.append(("ResNet18Scratch", resnet, test_loader_105, get_scores))

# Exp3 pretrained
base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feat = torch.nn.Sequential(*list(base.children())[:-1]).to(device)

def pretrained_forward(x):
    x = feat(x)
    x = x.view(x.size(0), -1)
    return F.normalize(x, dim=1)

models_list.append(("ResNet18_Pretrained", pretrained_forward, test_loader_224, get_scores_pretrained))


# ---------- ROC ----------

plt.figure(figsize=(8,6))
rows = []

for name, model, loader, scorer in models_list:
    print("Evaluating", name)

    scores, labels = scorer(model, loader)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    rows.append({"model": name, "auc": roc_auc})

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("All Models ROC")
plt.legend()
plt.grid()

plt.savefig("outputs/all_models_roc.png", dpi=300)

df = pd.DataFrame(rows)
df.to_csv("outputs/all_models_auc.csv", index=False)

print("\nAUC results:")
print(df)