import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from Data.datasets import PairDataset
from Models.models import KochBackbone
from losses.losses import contrastive_loss
from utils.logger import append_result

print("Contrastive training started")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_train = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTrain.txt"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

full_train_ds = PairDataset(pairs_train, root_dir, transform)

train_size = int(0.8 * len(full_train_ds))
val_size = len(full_train_ds) - train_size

generator = torch.Generator().manual_seed(42)

train_ds, val_ds = random_split(
    full_train_ds,
    [train_size, val_size],
    generator=generator
)

test_ds = PairDataset(pairs_test, root_dir, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print("Train pairs:", len(train_ds))
print("Val pairs:", len(val_ds))
print("Test pairs:", len(test_ds))
print("Device:", device)


@torch.no_grad()
def evaluate_distance(backbone, loader, device, threshold):
    backbone.eval()

    correct = 0
    total = 0

    all_distances = []
    all_labels = []

    for img1, img2, label in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        z1 = backbone(img1)
        z2 = backbone(img2)

        distances = F.pairwise_distance(z1, z2, p=2)

        # same identity if distance is small
        preds = (distances <= threshold).float()

        correct += (preds == label).sum().item()
        total += label.size(0)

        all_distances.append(distances.cpu())
        all_labels.append(label.cpu())

    accuracy = correct / total

    return accuracy, torch.cat(all_distances), torch.cat(all_labels)


def find_best_distance_threshold(distances, labels):
    best_threshold = 0.0
    best_accuracy = 0.0

    min_d = distances.min().item()
    max_d = distances.max().item()

    for threshold in torch.linspace(min_d, max_d, steps=101):
        preds = (distances <= threshold).float()
        accuracy = (preds == labels).float().mean().item()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold.item()

    return best_threshold, best_accuracy


checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

backbone = KochBackbone(embedding_dim=128).to(device)

optimizer = torch.optim.Adam(
    backbone.parameters(),
    lr=1e-4
)

num_epochs = 3
margins = [0.5, 1.0, 1.5]

overall_best = {
    "margin": None,
    "epoch": -1,
    "val_acc": 0.0,
    "threshold": None,
    "test_acc": None,
}

for margin in margins:
    print(f"\n=== Training contrastive model with margin = {margin} ===")

    backbone = KochBackbone(embedding_dim=128).to(device)

    optimizer = torch.optim.Adam(
        backbone.parameters(),
        lr=1e-4
    )

    best_val_acc = 0.0
    best_epoch = -1
    best_threshold_saved = None
    best_test_acc_saved = None

    for epoch in range(num_epochs):
        backbone.train()
        total_loss = 0.0

        for img1, img2, label in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            z1 = backbone(img1)
            z2 = backbone(img2)

            loss = contrastive_loss(z1, z2, label, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_default_acc, val_distances, val_labels = evaluate_distance(
            backbone, val_loader, device, threshold=margin / 2
        )

        best_threshold, val_best_acc = find_best_distance_threshold(
            val_distances, val_labels
        )

        test_acc, _, _ = evaluate_distance(
            backbone, test_loader, device, threshold=best_threshold
        )

        if val_best_acc > best_val_acc:
            best_val_acc = val_best_acc
            best_epoch = epoch + 1
            best_threshold_saved = best_threshold
            best_test_acc_saved = test_acc

            torch.save(
                {
                    "model_name": "contrastive_koch",
                    "epoch": best_epoch,
                    "embedding_dim": 128,
                    "margin": margin,
                    "backbone_state_dict": backbone.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_best_acc": best_val_acc,
                    "val_best_threshold": best_threshold,
                    "test_acc_at_val_threshold": test_acc,
                },
                checkpoint_dir / f"best_contrastive_margin_{margin}.pt"
            )

        print(
            f"Margin {margin} | "
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss = {avg_loss:.4f} | "
            f"val@default = {val_default_acc:.4f} | "
            f"val_best_thr = {best_threshold:.4f} | "
            f"val_best_acc = {val_best_acc:.4f} | "
            f"test_acc = {test_acc:.4f}"
        )

    print(f"Best epoch for margin {margin}: {best_epoch}")
    print(f"Best val acc for margin {margin}: {best_val_acc:.4f}")
    print(f"Best threshold for margin {margin}: {best_threshold_saved:.4f}")
    print(f"Test acc at best threshold: {best_test_acc_saved:.4f}")

    if best_val_acc > overall_best["val_acc"]:
        overall_best = {
            "margin": margin,
            "epoch": best_epoch,
            "val_acc": best_val_acc,
            "threshold": best_threshold_saved,
            "test_acc": best_test_acc_saved,
        }

        torch.save(
            {
                "model_name": "contrastive_koch_best_overall",
                "epoch": best_epoch,
                "embedding_dim": 128,
                "margin": margin,
                "backbone_state_dict": backbone.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_best_acc": best_val_acc,
                "val_best_threshold": best_threshold_saved,
                "test_acc_at_val_threshold": best_test_acc_saved,
            },
            checkpoint_dir / "best_contrastive.pt"
        )

print("\n=== Overall best contrastive model ===")
print(f"Margin: {overall_best['margin']}")
print(f"Epoch: {overall_best['epoch']}")
print(f"Validation accuracy: {overall_best['val_acc']:.4f}")
print(f"Threshold: {overall_best['threshold']:.4f}")
print(f"Test accuracy: {overall_best['test_acc']:.4f}")
print("Saved checkpoint: checkpoints/best_contrastive.pt")

append_result(
    "results_exp1.csv",
    {
        "method": "BCE_L1",
        "backbone": "KochCNN",
        "loss": "BCE",
        "margin": "",
        "selection_metric": "validation_accuracy",
        "best_epoch": best_epoch,
        "val_threshold": best_threshold_saved,
        "val_accuracy": best_val_acc,
        "test_accuracy": best_test_acc_saved,
        "checkpoint": "checkpoints/best_bce_l1.pt",
    }
)