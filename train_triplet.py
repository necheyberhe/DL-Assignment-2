import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.logger import append_result
from Data.datasets import (
    PairDataset,
    IdentityDataset,
    identities_from_pairs,
    BalancedIdentityBatchSampler,
)

from Models.models import KochBackbone
from losses.losses import triplet_loss, semihard_triplets

print("Triplet training started")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_train = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTrain.txt"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Pair datasets for validation/test evaluation
full_pair_train_ds = PairDataset(pairs_train, root_dir, transform)

train_size = int(0.8 * len(full_pair_train_ds))
val_size = len(full_pair_train_ds) - train_size

generator = torch.Generator().manual_seed(42)

pair_train_ds, pair_val_ds = random_split(
    full_pair_train_ds,
    [train_size, val_size],
    generator=generator
)

pair_test_ds = PairDataset(pairs_test, root_dir, transform)

val_loader = DataLoader(pair_val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(pair_test_ds, batch_size=32, shuffle=False)

# Identity dataset for triplet training
train_names = identities_from_pairs(full_pair_train_ds.pairs)

identity_ds = IdentityDataset(
    root_dir=root_dir,
    allowed_names=train_names,
    transform=transform
)

balanced_sampler = BalancedIdentityBatchSampler(
    identity_ds,
    identities_per_batch=16,
    images_per_identity=4
)

identity_loader = DataLoader(
    identity_ds,
    batch_sampler=balanced_sampler
)
print("Triplet train images:", len(identity_ds))
print("Train identities:", len(identity_ds.name_to_label))
print("Val pairs:", len(pair_val_ds))
print("Test pairs:", len(pair_test_ds))
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


def make_random_triplets(embeddings, labels):
    """
    Random triplet baseline.
    Requires at least two samples from some identities in the batch.
    """
    device = embeddings.device
    labels = labels.view(-1)

    anchors = []
    positives = []
    negatives = []

    batch_size = embeddings.size(0)

    for i in range(batch_size):
        same = torch.where(labels == labels[i])[0]
        diff = torch.where(labels != labels[i])[0]

        same = same[same != i]

        if len(same) == 0 or len(diff) == 0:
            continue

        p_idx = same[torch.randint(len(same), (1,), device=device)].item()
        n_idx = diff[torch.randint(len(diff), (1,), device=device)].item()

        anchors.append(i)
        positives.append(p_idx)
        negatives.append(n_idx)

    if len(anchors) == 0:
        return None

    anchors = torch.tensor(anchors, dtype=torch.long, device=device)
    positives = torch.tensor(positives, dtype=torch.long, device=device)
    negatives = torch.tensor(negatives, dtype=torch.long, device=device)

    return (
        embeddings.index_select(0, anchors),
        embeddings.index_select(0, positives),
        embeddings.index_select(0, negatives),
    )


def train_triplet_model(mode, margin=0.2, num_epochs=3):
    assert mode in {"random", "semihard"}

    print(f"\n=== Training triplet model: mode={mode}, margin={margin} ===")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

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
        used_batches = 0
        skipped_batches = 0

        for images, labels in identity_loader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = backbone(images)

            if mode == "random":
                triplets = make_random_triplets(embeddings, labels)
            else:
                triplets = semihard_triplets(embeddings, labels, margin=margin)

            if triplets is None:
                skipped_batches += 1
                continue

            anchor, positive, negative = triplets
            loss = triplet_loss(anchor, positive, negative, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            used_batches += 1

        if used_batches == 0:
            avg_loss = float("nan")
        else:
            avg_loss = total_loss / used_batches

        val_default_acc, val_distances, val_labels = evaluate_distance(
            backbone, val_loader, device, threshold=margin
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
                    "model_name": f"triplet_{mode}_koch",
                    "epoch": best_epoch,
                    "embedding_dim": 128,
                    "margin": margin,
                    "mode": mode,
                    "backbone_state_dict": backbone.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_best_acc": best_val_acc,
                    "val_best_threshold": best_threshold_saved,
                    "test_acc_at_val_threshold": best_test_acc_saved,
                },
                checkpoint_dir / f"best_triplet_{mode}_margin_{margin}.pt"            )

        print(
            f"Mode {mode} | "
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss = {avg_loss:.4f} | "
            f"used_batches = {used_batches} | "
            f"skipped_batches = {skipped_batches} | "
            f"val@margin = {val_default_acc:.4f} | "
            f"val_best_thr = {best_threshold:.4f} | "
            f"val_best_acc = {val_best_acc:.4f} | "
            f"test_acc = {test_acc:.4f}"
        )

    print(f"\nBest {mode} triplet epoch: {best_epoch}")
    print(f"Best {mode} validation accuracy: {best_val_acc:.4f}")
    print(f"Best {mode} validation threshold: {best_threshold_saved:.4f}")
    print(f"{mode} test accuracy at best threshold: {best_test_acc_saved:.4f}")
    print(f"Saved checkpoint: checkpoints/best_triplet_{mode}.pt")
    return {
        "mode": mode,
        "margin": margin,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_threshold": best_threshold_saved,
        "test_acc": best_test_acc_saved,
    }

if __name__ == "__main__":
    num_epochs = 3

    # Random triplet ablation, fixed margin
    random_result = train_triplet_model(
        mode="random",
        margin=0.2,
        num_epochs=num_epochs
    )

    # Semi-hard margin tuning
    semihard_results = []

    for margin in [0.1, 0.2, 0.5]:
        result = train_triplet_model(
            mode="semihard",
            margin=margin,
            num_epochs=num_epochs
        )
        semihard_results.append(result)

    best_semihard = max(
        semihard_results,
        key=lambda r: r["best_val_acc"]
    )

    print("\n=== Triplet summary ===")
    print("Random triplet:", random_result)

    print("\nSemi-hard margin results:")
    for result in semihard_results:
        print(result)

    print("\nBest semi-hard triplet:")
    print(best_semihard)
    append_result(
        "results_exp1.csv",
        {
            "method": "Triplet_Random",
            "backbone": "KochCNN",
            "loss": "triplet_random",
            "margin": random_result["margin"],
            "selection_metric": "validation_accuracy",
            "best_epoch": random_result["best_epoch"],
            "val_threshold": random_result["best_threshold"],
            "val_accuracy": random_result["best_val_acc"],
            "test_accuracy": random_result["test_acc"],
            "checkpoint": f"checkpoints/best_triplet_random_margin_{random_result['margin']}.pt",
        }
    )

    append_result(
        "results_exp1.csv",
        {
            "method": "Triplet_Semihard",
            "backbone": "KochCNN",
            "loss": "triplet_semihard",
            "margin": best_semihard["margin"],
            "selection_metric": "validation_accuracy",
            "best_epoch": best_semihard["best_epoch"],
            "val_threshold": best_semihard["best_threshold"],
            "val_accuracy": best_semihard["best_val_acc"],
            "test_accuracy": best_semihard["test_acc"],
            "checkpoint": f"checkpoints/best_triplet_semihard_margin_{best_semihard['margin']}.pt",
        }
    )