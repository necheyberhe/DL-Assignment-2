import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from Data.datasets import PairDataset
from Models.models import KochBackbone, SiameseBCEHead
from losses.losses import bce_l1_loss

from utils.logger import append_result
print("Script started")
import time
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
from utils.seed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)
seed = args.seed


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
def evaluate_bce(backbone, head, loader, device, threshold=0.5):
    backbone.eval()
    head.eval()

    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for img1, img2, label in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        z1 = backbone(img1)
        z2 = backbone(img2)
        logits = head(z1, z2)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        correct += (preds == label).sum().item()
        total += label.size(0)

        all_probs.append(probs.cpu())
        all_labels.append(label.cpu())

    accuracy = correct / total

    return accuracy, torch.cat(all_probs), torch.cat(all_labels)

@torch.no_grad()
def evaluate_bce_val_loss(backbone, head, loader, device):
    backbone.eval()
    head.eval()

    total_loss = 0.0

    for img1, img2, label in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        z1 = backbone(img1)
        z2 = backbone(img2)
        logits = head(z1, z2)

        loss = bce_l1_loss(logits, label)
        total_loss += loss.item()

    return total_loss / len(loader)


def find_best_threshold(probs, labels):
    best_threshold = 0.5
    best_accuracy = 0.0

    for threshold in torch.linspace(0.0, 1.0, steps=101):
        preds = (probs >= threshold).float()
        accuracy = (preds == labels).float().mean().item()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold.item()

    return best_threshold, best_accuracy


checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

backbone = KochBackbone(embedding_dim=128).to(device)
head = SiameseBCEHead(embedding_dim=128).to(device)

optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(head.parameters()),
    lr=1e-4
)

num_epochs = 30

best_val_acc = 0.0
best_epoch = -1
best_threshold_saved = None
best_test_acc_saved = None

patience = 5
min_delta = 1e-4
epochs_without_improvement = 0
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

history = []
start_time = time.time()
for epoch in range(num_epochs):
    backbone.train()
    head.train()

    total_loss = 0.0

    for img1, img2, label in train_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        z1 = backbone(img1)
        z2 = backbone(img2)
        logits = head(z1, z2)

        loss = bce_l1_loss(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_acc_05, val_probs, val_labels = evaluate_bce(
        backbone, head, val_loader, device, threshold=0.5
    )

    best_threshold, val_best_acc = find_best_threshold(
        val_probs, val_labels
    )

    test_acc, _, _ = evaluate_bce(
        backbone, head, test_loader, device, threshold=best_threshold
    )
    val_loss = evaluate_bce_val_loss(
        backbone, head, val_loader, device
    )

    elapsed = time.time() - start_time

    history.append({
        "seed": seed,
        "model": "BCE_L1",
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_loss": val_loss,
        "val_acc_at_0_5": val_acc_05,
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
                "model_name": "bce_l1_koch",
                "epoch": best_epoch,
                "embedding_dim": 128,
                "backbone_state_dict": backbone.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_best_acc": best_val_acc,
                "val_best_threshold": best_threshold,
                "test_acc_at_val_threshold": test_acc,
            },
            # checkpoint_dir / "best_bce_l1.pt"
            checkpoint_dir / f"best_bce_l1_seed_{seed}.pt"
        )
    else:
        epochs_without_improvement += 1
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"loss = {avg_loss:.4f} | "
        f"val@0.5 = {val_acc_05:.4f} | "
        f"val_best_thr = {best_threshold:.2f} | "
        f"val_best_acc = {val_best_acc:.4f} | "
        f"test_acc = {test_acc:.4f}"
    )
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}")
        break
    

print(f"Best BCE epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best validation threshold: {best_threshold_saved:.2f}")
print(f"Test accuracy at best validation threshold: {best_test_acc_saved:.4f}")
print(f"Saved checkpoint: checkpoints/best_bce_l1_seed_{seed}.pt")
history_df = pd.DataFrame(history)
history_path = output_dir / f"history_bce_l1_seed_{seed}.csv"
history_df.to_csv(history_path, index=False)

print(f"Saved history: {history_path}")

append_result(
    f"results_exp1_seed_{seed}.csv",
    {
        "experiment": "exp1",
        "method": "BCE_L1",
        "backbone": "KochCNN",
        "loss": "bce_l1",
        "margin": "",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_threshold": best_threshold_saved,
        "val_accuracy": best_val_acc,
        "test_accuracy": best_test_acc_saved,
        "checkpoint": f"checkpoints/best_bce_l1_seed_{seed}.pt",
        "history_csv": str(history_path),
        "train_time_sec": time.time() - start_time,
        "max_epochs": num_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "stopping_rule": "validation_accuracy",
    }
)