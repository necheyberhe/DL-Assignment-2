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
from Models.resnet import ResNet18Backbone
from losses.losses import triplet_loss, semihard_triplets
from utils.logger import append_result
from utils.seed import set_seed


print("Experiment 2: Backbone comparison started")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--only_backbone", type=str, default=None)
args = parser.parse_args()

seed = args.seed
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_train = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTrain.txt"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

embedding_dim = 128
margin = 0.2
num_epochs = 30
patience = 5
min_delta = 1e-4
lr = 1e-4

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

full_pair_train_ds = PairDataset(pairs_train, root_dir, transform)

train_size = int(0.8 * len(full_pair_train_ds))
val_size = len(full_pair_train_ds) - train_size

generator = torch.Generator().manual_seed(42)

pair_train_ds, pair_val_ds = random_split(
    full_pair_train_ds,
    [train_size, val_size],
    generator=generator,
)

pair_test_ds = PairDataset(pairs_test, root_dir, transform)

val_loader = DataLoader(pair_val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(pair_test_ds, batch_size=32, shuffle=False)

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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_macs(model, input_size=(1, 3, 105, 105)):
    try:
        from thop import profile
    except ImportError:
        print("thop not installed. Run: pip install thop")
        return None

    model.eval()
    dummy = torch.randn(*input_size).to(device)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)

    return macs


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


@torch.no_grad()
def evaluate_triplet_validation_loss(backbone):
    backbone.eval()

    total_loss = 0.0
    used_batches = 0
    skipped_batches = 0

    for images, labels in val_identity_loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = backbone(images)

        triplets = semihard_triplets(
            embeddings,
            labels,
            margin=margin,
        )

        if triplets is None:
            skipped_batches += 1
            continue

        anchor, positive, negative = triplets
        loss = triplet_loss(anchor, positive, negative, margin=margin)

        total_loss += loss.item()
        used_batches += 1

    val_loss = total_loss / used_batches if used_batches > 0 else float("nan")

    return val_loss, used_batches, skipped_batches


def make_backbone(backbone_name):
    if backbone_name == "koch":
        return KochBackbone(embedding_dim=embedding_dim).to(device)

    if backbone_name == "resnet18":
        return ResNet18Backbone(embedding_dim=embedding_dim).to(device)

    raise ValueError(f"Unknown backbone: {backbone_name}")


def train_backbone(backbone_name):
    print(f"\n=== Training backbone: {backbone_name} ===")

    backbone = make_backbone(backbone_name)

    params = count_params(backbone)
    macs = compute_macs(backbone)

    print(f"Parameters: {params:,}")
    if macs is not None:
        print(f"MACs: {macs:,}")
        print(f"Approx FLOPs (2 x MACs): {2 * macs:,}")

    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)

    best_val_acc = 0.0
    best_epoch = -1
    best_threshold_saved = None
    best_test_acc_saved = None
    epochs_without_improvement = 0

    start_time = time.time()
    history = []

    checkpoint_path = (
        checkpoint_dir / f"exp2_best_{backbone_name}_seed_{seed}.pt"
    )

    for epoch in range(num_epochs):
        backbone.train()

        total_loss = 0.0
        used_batches = 0
        skipped_batches = 0

        for images, labels in identity_loader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = backbone(images)

            triplets = semihard_triplets(
                embeddings,
                labels,
                margin=margin,
            )

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
            evaluate_triplet_validation_loss(backbone)
        )

        elapsed = time.time() - start_time

        history.append({
            "seed": seed,
            "backbone": backbone_name,
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

            torch.save(
                {
                    "experiment": "exp2_backbone",
                    "backbone": backbone_name,
                    "epoch": best_epoch,
                    "embedding_dim": embedding_dim,
                    "loss": "triplet_semihard",
                    "margin": margin,
                    "params": params,
                    "macs": macs,
                    "approx_flops": None if macs is None else 2 * macs,
                    "input_size": "1x3x105x105",
                    "flops_convention": "FLOPs = 2 * MACs",
                    "backbone_state_dict": backbone.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_best_acc": best_val_acc,
                    "val_best_threshold": best_threshold_saved,
                    "test_acc_at_val_threshold": best_test_acc_saved,
                    "seed": seed,
                    "max_epochs": num_epochs,
                    "patience": patience,
                    "min_delta": min_delta,
                    "stopping_rule": "validation_accuracy",
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Backbone {backbone_name} | "
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
            print(
                f"Early stopping for {backbone_name} "
                f"at epoch {epoch + 1}; best epoch was {best_epoch}"
            )
            break

    training_time_sec = time.time() - start_time

    history_path = output_dir / f"history_exp2_{backbone_name}_seed_{seed}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    print(f"\nBest epoch for {backbone_name}: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation threshold: {best_threshold_saved:.4f}")
    print(f"Test accuracy at best threshold: {best_test_acc_saved:.4f}")
    print(f"Training time: {training_time_sec:.2f} seconds")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")

    append_result(
        f"results_exp2_seed_{seed}.csv",
        {
            "experiment": "exp2",
            "backbone": backbone_name,
            "loss": "triplet_semihard",
            "margin": margin,
            "embedding_dim": embedding_dim,
            "params": params,
            "macs": macs,
            "approx_flops": None if macs is None else 2 * macs,
            "input_size": "1x3x105x105",
            "flops_convention": "FLOPs = 2 * MACs",
            "train_time_sec": training_time_sec,
            "best_epoch": best_epoch,
            "val_threshold": best_threshold_saved,
            "val_accuracy": best_val_acc,
            "test_accuracy": best_test_acc_saved,
            "checkpoint": str(checkpoint_path),
            "history_csv": str(history_path),
            "seed": seed,
            "max_epochs": num_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "stopping_rule": "validation_accuracy",
        },
    )

    return {
        "backbone": backbone_name,
        "params": params,
        "macs": macs,
        "approx_flops": None if macs is None else 2 * macs,
        "train_time_sec": training_time_sec,
        "best_epoch": best_epoch,
        "val_threshold": best_threshold_saved,
        "val_accuracy": best_val_acc,
        "test_accuracy": best_test_acc_saved,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
    }


if __name__ == "__main__":
    results = []

    backbones = ["koch", "resnet18"]

    if args.only_backbone is not None:
        if args.only_backbone not in {"koch", "resnet18"}:
            raise ValueError("--only_backbone must be one of: koch, resnet18")
        backbones = [args.only_backbone]

    for backbone_name in backbones:
        result = train_backbone(backbone_name)
        results.append(result)

    print("\n=== Experiment 2 summary ===")
    for result in results:
        print(result)