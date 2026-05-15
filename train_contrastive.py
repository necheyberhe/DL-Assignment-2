import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from Data.datasets import PairDataset
from Models.models import KochBackbone
from losses.losses import contrastive_loss
from utils.logger import append_result
from utils.seed import set_seed


print("Contrastive training started")

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--margin",
    type=float,
    default=1.0
)

args = parser.parse_args()

seed = args.seed
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_train = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTrain.txt"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

embedding_dim = 128
num_epochs = 30
margin = args.margin
patience = 5
min_delta = 1e-4
lr = 1e-4
batch_size = 32

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

full_train_ds = PairDataset(pairs_train, root_dir, transform)

train_size = int(0.8 * len(full_train_ds))
val_size = len(full_train_ds) - train_size

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(
    full_train_ds,
    [train_size, val_size],
    generator=generator,
)

test_ds = PairDataset(pairs_test, root_dir, transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print("Seed:", seed)
print("Train pairs:", len(train_ds))
print("Val pairs:", len(val_ds))
print("Test pairs:", len(test_ds))
print("Device:", device)


@torch.no_grad()
def evaluate_distance(backbone, loader, threshold):
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
        preds = (distances <= threshold).float()

        correct += (preds == label).sum().item()
        total += label.size(0)

        all_distances.append(distances.cpu())
        all_labels.append(label.cpu())

    accuracy = correct / total
    return accuracy, torch.cat(all_distances), torch.cat(all_labels)


@torch.no_grad()
def evaluate_contrastive_loss(backbone, loader):
    backbone.eval()

    total_loss = 0.0

    for img1, img2, label in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        z1 = backbone(img1)
        z2 = backbone(img2)

        loss = contrastive_loss(z1, z2, label, margin=margin)
        total_loss += loss.item()

    return total_loss / len(loader)


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


backbone = KochBackbone(embedding_dim=embedding_dim).to(device)
optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)

best_val_acc = 0.0
best_epoch = -1
best_threshold_saved = None
best_test_acc_saved = None
epochs_without_improvement = 0

history = []
start_time = time.time()

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

    train_loss = total_loss / len(train_loader)

    val_default_acc, val_distances, val_labels = evaluate_distance(
        backbone,
        val_loader,
        threshold=margin / 2,
    )

    best_threshold, val_best_acc = find_best_distance_threshold(
        val_distances,
        val_labels,
    )

    val_loss = evaluate_contrastive_loss(backbone, val_loader)

    test_acc, _, _ = evaluate_distance(
        backbone,
        test_loader,
        threshold=best_threshold,
    )

    elapsed = time.time() - start_time

    history.append({
        "seed": seed,
        "model": "Contrastive",
        "margin": margin,
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_default_acc": val_default_acc,
        "val_best_acc": val_best_acc,
        "val_best_threshold": best_threshold,
        "test_acc_at_val_threshold": test_acc,
        "wall_time_sec": elapsed,
    })

    if val_best_acc > best_val_acc + min_delta:
        best_val_acc = val_best_acc
        best_epoch = epoch + 1
        best_threshold_saved = best_threshold
        best_test_acc_saved = test_acc
        epochs_without_improvement = 0

        torch.save(
            {
                "model_name": "contrastive_koch",
                "seed": seed,
                "epoch": best_epoch,
                "embedding_dim": embedding_dim,
                "margin": margin,
                "max_epochs": num_epochs,
                "patience": patience,
                "min_delta": min_delta,
                "stopping_rule": "validation_accuracy",
                "backbone_state_dict": backbone.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_best_acc": best_val_acc,
                "val_best_threshold": best_threshold_saved,
                "test_acc_at_val_threshold": best_test_acc_saved,
            },
            checkpoint_dir / f"best_contrastive_margin_{margin}_seed_{seed}.pt",
        )
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val@default={val_default_acc:.4f} | "
        f"val_best_thr={best_threshold:.4f} | "
        f"val_best_acc={val_best_acc:.4f} | "
        f"test_acc={test_acc:.4f}"
    )

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}")
        break

train_time_sec = time.time() - start_time

history_path = output_dir / f"history_contrastive_margin_{margin}_seed_{seed}.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

result_path = f"results_exp1_seed_{seed}.csv"

append_result(
    f"results_exp1_seed_{seed}.csv",
    {
        "experiment": "exp1",
        "method": "Contrastive",
        "backbone": "KochCNN",
        "loss": "contrastive",
        "margin": margin,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_threshold": best_threshold_saved,
        "val_accuracy": best_val_acc,
        "test_accuracy": best_test_acc_saved,
        "checkpoint": f"checkpoints/best_contrastive_margin_{margin}_seed_{seed}.pt",
        "history_csv": str(history_path),
        "train_time_sec": train_time_sec,
        "max_epochs": num_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "stopping_rule": "validation_accuracy",
    },
)

print("\n=== Contrastive result ===")
print(f"Seed: {seed}")
print(f"Margin: {margin}")
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best threshold: {best_threshold_saved:.4f}")
print(f"Test accuracy: {best_test_acc_saved:.4f}")
print(f"Training time: {train_time_sec:.2f} seconds")
print(f"Saved checkpoint: checkpoints/best_contrastive_seed_{seed}.pt")
print(f"Saved history: {history_path}")
print(f"Saved result: {result_path}")