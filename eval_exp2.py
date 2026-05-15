import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms

from Data.datasets import PairDataset
from Models.models import KochBackbone
from Models.resnet import ResNet18Backbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

results_csv = "results_exp2_seed_42.csv"
results_df = pd.read_csv(results_csv)

models = []

for _, row in results_df.iterrows():

    backbone_name = row["backbone"]

    if backbone_name == "koch":
        display_name = "KochCNN"

    elif backbone_name == "resnet18":
        display_name = "ResNet18Scratch"

    else:
        raise ValueError(backbone_name)

    models.append(
        (
            display_name,
            backbone_name,
            row["checkpoint"],
        )
    )

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_ds = PairDataset(pairs_test, root_dir, transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


def load_exp2_backbone(backbone_name, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if backbone_name == "koch":
        model = KochBackbone(embedding_dim=128).to(device)
    elif backbone_name == "resnet18":
        model = ResNet18Backbone(embedding_dim=128).to(device)
    else:
        raise ValueError(backbone_name)

    state = checkpoint["backbone_state_dict"]

    clean_state = {
        k: v for k, v in state.items()
        if "total_ops" not in k and "total_params" not in k
    }

    model.load_state_dict(clean_state, strict=False)
    model.eval()

    return model

@torch.no_grad()
def get_scores(model):
    scores = []
    labels = []

    for img1, img2, label in test_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        z1 = model(img1)
        z2 = model(img2)

        distances = F.pairwise_distance(z1, z2, p=2)
        similarity = -distances

        scores.append(similarity.cpu())
        labels.append(label.cpu())

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


# models = [
#     ("KochCNN", "koch", "checkpoints/exp2_best_koch.pt"),
#     ("ResNet18Scratch", "resnet18", "checkpoints/exp2_best_resnet18.pt"),
# ]

rows = []

plt.figure(figsize=(8, 6))

for display_name, backbone_name, ckpt in models:
    print(f"Evaluating {display_name}")

    model = load_exp2_backbone(backbone_name, ckpt)
    scores, labels = get_scores(model)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    safe_name = display_name.lower().replace(" ", "_").replace("-", "_")

    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
    }).to_csv(output_dir / f"roc_exp2_{safe_name}.csv", index=False)
        
    rows.append({
        "backbone": display_name,
        "auc": roc_auc,
        "checkpoint": ckpt,
    })

    plt.plot(fpr, tpr, label=f"{display_name} (AUC={roc_auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Experiment 2 ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = output_dir / "exp2_roc_curves.png"
auc_path = output_dir / "exp2_auc.csv"

plt.savefig(roc_path, dpi=300)
plt.close()

df = pd.DataFrame(rows)
df.to_csv(auc_path, index=False)

print(df)
print(f"Saved ROC plot to: {roc_path}")
print(f"Saved AUC table to: {auc_path}")