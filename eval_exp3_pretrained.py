import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json

from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader

from Data.datasets import PairDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

episodes_path = Path("outputs/oneshot_episodes.json")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


# IMPORTANT: ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ResNet requirement
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------- Model ----------

class FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.feature_extractor = nn.Sequential(
            *list(base.children())[:-1]
        )

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return F.normalize(x, p=2, dim=1)


model = FrozenResNet18().to(device)
model.eval()


# ---------- Pair evaluation ----------

test_ds = PairDataset(pairs_test, root_dir, transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


@torch.no_grad()
def get_scores():
    scores = []
    labels = []

    for img1, img2, label in test_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        z1 = model(img1)
        z2 = model(img2)

        cos_sim = F.cosine_similarity(z1, z2)

        scores.append(cos_sim.cpu())
        labels.append(label.cpu())

    return torch.cat(scores), torch.cat(labels)


def find_best_threshold(scores, labels):
    best_thr = 0.0
    best_acc = 0.0

    for t in torch.linspace(-1, 1, 200):
        preds = (scores >= t).float()
        acc = (preds == labels).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_thr = t.item()

    return best_thr, best_acc


scores, labels = get_scores()
best_thr, test_acc = find_best_threshold(scores, labels)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(labels.numpy(), scores.numpy())
roc_auc = auc(fpr, tpr)

pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
}).to_csv(output_dir / "roc_exp3_frozen_resnet18.csv", index=False)

print("AUC:", roc_auc)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Frozen ImageNet ResNet-18 (AUC={roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Experiment 3 ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "exp3_roc_curve.png", dpi=400)
plt.close()
print("Pretrained baseline:")
print("Best threshold:", best_thr)
print("Test accuracy:", test_acc)


# ---------- One-shot ----------

def load_image(path):
    return transform(Image.open(path).convert("RGB"))


@torch.no_grad()
def eval_oneshot():
    with open(episodes_path, "r") as f:
        episodes = json.load(f)

    results = {}

    for n, eps in episodes.items():
        correct = 0

        for ep in eps:
            query = load_image(ep["query_image"]).unsqueeze(0).to(device)

            support_imgs = [
                load_image(x["image"]) for x in ep["support"]
            ]
            support = torch.stack(support_imgs).to(device)

            query_batch = query.repeat(support.size(0), 1, 1, 1)

            zq = model(query_batch)
            zs = model(support)

            sims = F.cosine_similarity(zq, zs)

            pred = torch.argmax(sims).item()

            if pred == ep["correct_index"]:
                correct += 1

        results[int(n)] = correct / len(eps)

    return results


oneshot = eval_oneshot()

print("One-shot:", oneshot)


# ---------- Save ----------
df = pd.DataFrame([{
    "method": "ResNet18_Pretrained_Frozen",
    "test_accuracy": test_acc,
    "threshold": best_thr,
    "auc": roc_auc,
    "oneshot_N2": oneshot[2],
    "oneshot_N5": oneshot[5],
    "oneshot_N20": oneshot[20],
}])

df.to_csv(output_dir / "exp3_pretrained.csv", index=False)