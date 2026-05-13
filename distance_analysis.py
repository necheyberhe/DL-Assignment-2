import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from Data.datasets import IdentityDataset, PairDataset, identities_from_pairs
from Models.models import KochBackbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

print("Loading dataset...")

pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

test_pair_ds = PairDataset(
    pairs_test,
    root_dir,
    transform
)

test_names = identities_from_pairs(test_pair_ds.pairs)

dataset = IdentityDataset(
    root_dir=root_dir,
    allowed_names=test_names,
    transform=transform
)

# -------------------------------------------------
# Select identities
# -------------------------------------------------

random.seed(42)

identity_to_indices = defaultdict(list)

for idx, (_, label, name) in enumerate(dataset.samples):
    identity_to_indices[name].append(idx)

eligible = [
    name for name, idxs in identity_to_indices.items()
    if len(idxs) >= 5
]

selected_names = random.sample(eligible, 25)

selected_indices = []

for name in selected_names:
    selected_indices.extend(identity_to_indices[name][:10])

subset = Subset(dataset, selected_indices)

loader = DataLoader(
    subset,
    batch_size=64,
    shuffle=False
)

print("Selected identities:", len(selected_names))
print("Selected images:", len(selected_indices))

# -------------------------------------------------
# Load model
# -------------------------------------------------

print("Loading model...")

model = KochBackbone(embedding_dim=128).to(device)

checkpoint = torch.load(
    "checkpoints/exp2_best_koch.pt",
    map_location=device
)

state = checkpoint["backbone_state_dict"]

clean_state = {
    k: v for k, v in state.items()
    if "total_ops" not in k and "total_params" not in k
}

model.load_state_dict(clean_state, strict=False)

model.eval()

# -------------------------------------------------
# Extract embeddings
# -------------------------------------------------

print("Extracting embeddings...")

all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        emb = model(images)

        all_embeddings.append(emb.cpu())
        all_labels.extend(labels.numpy())

embeddings = torch.cat(all_embeddings)
labels = np.array(all_labels)

print("Embedding shape:", embeddings.shape)

# -------------------------------------------------
# Compute pairwise distances
# -------------------------------------------------

print("Computing distances...")

intra_dists = []
inter_dists = []

N = embeddings.size(0)

for i in range(N):
    for j in range(i + 1, N):

        d = F.pairwise_distance(
            embeddings[i].unsqueeze(0),
            embeddings[j].unsqueeze(0),
            p=2
        ).item()

        if labels[i] == labels[j]:
            intra_dists.append(d)
        else:
            inter_dists.append(d)

intra_dists = np.array(intra_dists)
inter_dists = np.array(inter_dists)

print("\nDistance statistics")
print("-------------------")
print(f"Intra-class mean: {intra_dists.mean():.4f}")
print(f"Intra-class std : {intra_dists.std():.4f}")

print(f"Inter-class mean: {inter_dists.mean():.4f}")
print(f"Inter-class std : {inter_dists.std():.4f}")

# -------------------------------------------------
# Plot histograms
# -------------------------------------------------

print("Creating histogram plot...")

plt.figure(figsize=(10, 6))

plt.hist(
    intra_dists,
    bins=50,
    alpha=0.6,
    density=True,
    label="Intra-class"
)

plt.hist(
    inter_dists,
    bins=50,
    alpha=0.6,
    density=True,
    label="Inter-class"
)


plt.axvline(
    intra_dists.mean(),
    linestyle="--",
    linewidth=2,
    label="Mean intra"
)

plt.axvline(
    inter_dists.mean(),
    linestyle="--",
    linewidth=2,
    label="Mean inter"
)


plt.xlabel("Euclidean Distance")
plt.ylabel("Density")
plt.title("Intra-class and Inter-class Embedding Distance Distributions")
plt.legend()
plt.grid(True)

output_path = "outputs/distance_distributions.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved histogram plot to: {output_path}")