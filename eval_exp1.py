import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms

from Data.datasets import PairDataset
from Models.models import KochBackbone, SiameseBCEHead


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

results_csv = "results_exp1.csv"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_ds = PairDataset(
    pairs_file=pairs_test,
    root_dir=root_dir,
    transform=transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False
)


@torch.no_grad()
def get_bce_scores(backbone, head, loader):
    backbone.eval()
    head.eval()

    scores = []
    labels = []

    for img1, img2, label in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        z1 = backbone(img1)
        z2 = backbone(img2)

        logits = head(z1, z2)
        probs = torch.sigmoid(logits)

        scores.append(probs.cpu())
        labels.append(label.cpu())

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def get_distance_scores(backbone, loader):
    """
    For ROC, higher score should mean more likely same identity.
    Distance is lower for same identity, so we use negative distance.
    """
    backbone.eval()

    scores = []
    labels = []

    for img1, img2, label in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        z1 = backbone(img1)
        z2 = backbone(img2)

        distances = F.pairwise_distance(z1, z2, p=2)
        similarity_scores = -distances

        scores.append(similarity_scores.cpu())
        labels.append(label.cpu())

    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def load_bce_checkpoint(path):
    checkpoint = torch.load(path, map_location=device)

    backbone = KochBackbone(embedding_dim=128).to(device)
    head = SiameseBCEHead(embedding_dim=128).to(device)

    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])

    return backbone, head


def load_backbone_checkpoint(path):
    checkpoint = torch.load(path, map_location=device)

    backbone = KochBackbone(embedding_dim=128).to(device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])

    return backbone


def main():
    results = pd.read_csv(results_csv)

    roc_rows = []

    plt.figure(figsize=(8, 6))

    for _, row in results.iterrows():
        method = row["method"]
        checkpoint_path = row["checkpoint"]

        print(f"Evaluating {method} from {checkpoint_path}")

        if method == "BCE_L1":
            backbone, head = load_bce_checkpoint(checkpoint_path)
            scores, labels = get_bce_scores(backbone, head, test_loader)
        else:
            backbone = load_backbone_checkpoint(checkpoint_path)
            scores, labels = get_distance_scores(backbone, test_loader)

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        roc_rows.append({
            "method": method,
            "auc": roc_auc
        })

        plt.plot(fpr, tpr, label=f"{method} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Experiment 1 ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    roc_path = output_dir / "exp1_roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    auc_df = pd.DataFrame(roc_rows)
    auc_path = output_dir / "exp1_auc.csv"
    auc_df.to_csv(auc_path, index=False)

    print("\nAUC results:")
    print(auc_df)

    print(f"\nSaved ROC plot to: {roc_path}")
    print(f"Saved AUC table to: {auc_path}")


if __name__ == "__main__":
    main()