import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from Data.datasets import (
    PairDataset,
    IdentityDataset,
    identities_from_pairs,
    BalancedIdentityBatchSampler,
)
from Models.models import KochBackbone
from losses.losses import triplet_loss, semihard_triplets
from utils.logger import append_result
from utils.seed import set_seed


print("Triplet training started")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed = args.seed
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_train = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTrain.txt"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

embedding_dim = 128
num_epochs = 30
batch_size_eval = 32
lr = 1e-4
patience = 5
min_delta = 1e-4

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# -------------------------
# Pair datasets
# -------------------------

full_pair_train_ds = PairDataset(pairs_train, root_dir, transform)

train_size = int(0.8 * len(full_pair_train_ds))
val_size = len(full_pair_train_ds) - train_size

# Fixed validation split for fair Experiment 1 comparison
generator = torch.Generator().manual_seed(42)

pair_train_ds, pair_val_ds = random_split(
    full_pair_train_ds,
    [train_size, val_size],
    generator=generator,
)

pair_test_ds = PairDataset(pairs_test, root_dir, transform)

val_loader = DataLoader(pair_val_ds, batch_size=batch_size_eval, shuffle=False)
test_loader = DataLoader(pair_test_ds, batch_size=batch_size_eval, shuffle=False)

# -------------------------
# Identity datasets
# -------------------------

train_names = identities_from_pairs(full_pair_train_ds.pairs)

val_pair_indices = pair_val_ds.indices
val_pairs = [full_pair_train_ds.pairs[i] for i in val_pair_indices]
val_names = identities_from_pairs(val_pairs)

identity_ds = IdentityDataset(
    root_dir=root_dir,
    allowed_names=train_names,
    transform=transform,
)

balanced_sampler = BalancedIdentityBatchSampler(
    identity_ds,
    identities_per_batch=16,
    images_per_identity=4,
)

identity_loader = DataLoader(
    identity_ds,
    batch_sampler=balanced_sampler,
)

val_identity_ds = IdentityDataset(
    root_dir=root_dir,
    allowed_names=val_names,
    transform=transform,
)

val_balanced_sampler = BalancedIdentityBatchSampler(
    val_identity_ds,
    identities_per_batch=16,
    images_per_identity=4,
)

val_identity_loader = DataLoader(
    val_identity_ds,
    batch_sampler=val_balanced_sampler,
)

print("Triplet train images:", len(identity_ds))
print("Train identities:", len(identity_ds.name_to_label))
print("Validation identity images:", len(val_identity_ds))
print("Validation identities:", len(val_identity_ds.name_to_label))
print("Val pairs:", len(pair_val_ds))
print("Test pairs:", len(pair_test_ds))
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

    return correct / total, torch.cat(all_distances), torch.cat(all_labels)


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
    device_local = embeddings.device
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

        p_idx = same[torch.randint(len(same), (1,), device=device_local)].item()
        n_idx = diff[torch.randint(len(diff), (1,), device=device_local)].item()

        anchors.append(i)
        positives.append(p_idx)
        negatives.append(n_idx)

    if len(anchors) == 0:
        return None

    anchors = torch.tensor(anchors, dtype=torch.long, device=device_local)
    positives = torch.tensor(positives, dtype=torch.long, device=device_local)
    negatives = torch.tensor(negatives, dtype=torch.long, device=device_local)

    return (
        embeddings.index_select(0, anchors),
        embeddings.index_select(0, positives),
        embeddings.index_select(0, negatives),
    )


@torch.no_grad()
def evaluate_triplet_validation_loss(backbone, mode, margin):
    backbone.eval()

    total_loss = 0.0
    used_batches = 0
    skipped_batches = 0

    for images, labels in val_identity_loader:
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

        total_loss += loss.item()
        used_batches += 1

    val_loss = total_loss / used_batches if used_batches > 0 else float("nan")
    return val_loss, used_batches, skipped_batches


def train_triplet_model(mode, margin):
    assert mode in {"random", "semihard"}

    print(f"\n=== Training triplet model: mode={mode}, margin={margin} ===")

    backbone = KochBackbone(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)

    best_val_acc = 0.0
    best_epoch = -1
    best_threshold_saved = None
    best_test_acc_saved = None
    checkpoint_path = None

    epochs_without_improvement = 0
    history = []
    start_time = time.time()

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

        train_loss = total_loss / used_batches if used_batches > 0 else float("nan")

        val_default_acc, val_distances, val_labels = evaluate_distance(
            backbone,
            val_loader,
            threshold=margin,
        )

        best_threshold, val_best_acc = find_best_distance_threshold(
            val_distances,
            val_labels,
        )

        test_acc, _, _ = evaluate_distance(
            backbone,
            test_loader,
            threshold=best_threshold,
        )

        val_loss, val_loss_used_batches, val_loss_skipped_batches = (
            evaluate_triplet_validation_loss(
                backbone=backbone,
                mode=mode,
                margin=margin,
            )
        )

        elapsed = time.time() - start_time

        history.append({
            "seed": seed,
            "model": f"Triplet_{mode}",
            "margin": margin,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_default_acc": val_default_acc,
            "val_best_acc": val_best_acc,
            "val_best_threshold": best_threshold,
            "test_acc_at_val_threshold": test_acc,
            "used_batches": used_batches,
            "skipped_batches": skipped_batches,
            "val_loss_used_batches": val_loss_used_batches,
            "val_loss_skipped_batches": val_loss_skipped_batches,
            "wall_time_sec": elapsed,
        })

        if val_best_acc > best_val_acc + min_delta:
            best_val_acc = val_best_acc
            best_epoch = epoch + 1
            best_threshold_saved = best_threshold
            best_test_acc_saved = test_acc
            epochs_without_improvement = 0

            checkpoint_path = (
                checkpoint_dir
                / f"best_triplet_{mode}_margin_{margin}_seed_{seed}.pt"
            )

            torch.save(
                {
                    "model_name": f"triplet_{mode}_koch",
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
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Mode {mode} | "
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"used_batches={used_batches} | "
            f"skipped_batches={skipped_batches} | "
            f"val_loss_batches={val_loss_used_batches} | "
            f"val@margin={val_default_acc:.4f} | "
            f"val_best_thr={best_threshold:.4f} | "
            f"val_best_acc={val_best_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}")
            break

    train_time_sec = time.time() - start_time

    history_path = (
        output_dir / f"history_triplet_{mode}_margin_{margin}_seed_{seed}.csv"
    )
    pd.DataFrame(history).to_csv(history_path, index=False)

    print(f"\nBest {mode} triplet epoch: {best_epoch}")
    print(f"Best {mode} validation accuracy: {best_val_acc:.4f}")
    print(f"Best {mode} validation threshold: {best_threshold_saved:.4f}")
    print(f"{mode} test accuracy at best threshold: {best_test_acc_saved:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")

    return {
        "mode": mode,
        "margin": margin,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_threshold": best_threshold_saved,
        "test_acc": best_test_acc_saved,
        "checkpoint": str(checkpoint_path),
        "train_time_sec": train_time_sec,
        "history_csv": str(history_path),
    }


if __name__ == "__main__":
    random_result = train_triplet_model(
        mode="random",
        margin=0.2,
    )

    semihard_results = []

    for margin in [0.1, 0.2, 0.5]:
        result = train_triplet_model(
            mode="semihard",
            margin=margin,
        )
        semihard_results.append(result)

    best_semihard = max(
        semihard_results,
        key=lambda r: r["best_val_acc"],
    )

    print("\n=== Triplet summary ===")
    print("Random triplet:", random_result)

    print("\nSemi-hard margin results:")
    for result in semihard_results:
        print(result)

    print("\nBest semi-hard triplet:")
    print(best_semihard)

    result_path = f"results_exp1_seed_{seed}.csv"

    append_result(
        result_path,
        {
            "experiment": "exp1",
            "method": "Triplet_Random",
            "backbone": "KochCNN",
            "loss": "triplet_random",
            "margin": random_result["margin"],
            "seed": seed,
            "best_epoch": random_result["best_epoch"],
            "val_threshold": random_result["best_threshold"],
            "val_accuracy": random_result["best_val_acc"],
            "test_accuracy": random_result["test_acc"],
            "checkpoint": random_result["checkpoint"],
            "history_csv": random_result["history_csv"],
            "train_time_sec": random_result["train_time_sec"],
            "max_epochs": num_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "stopping_rule": "validation_accuracy",
        },
    )

    append_result(
        result_path,
        {
            "experiment": "exp1",
            "method": "Triplet_Semihard",
            "backbone": "KochCNN",
            "loss": "triplet_semihard",
            "margin": best_semihard["margin"],
            "seed": seed,
            "best_epoch": best_semihard["best_epoch"],
            "val_threshold": best_semihard["best_threshold"],
            "val_accuracy": best_semihard["best_val_acc"],
            "test_accuracy": best_semihard["test_acc"],
            "checkpoint": best_semihard["checkpoint"],
            "history_csv": best_semihard["history_csv"],
            "train_time_sec": best_semihard["train_time_sec"],
            "max_epochs": num_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "stopping_rule": "validation_accuracy",
        },
    )