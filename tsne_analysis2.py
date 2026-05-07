import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from Data.datasets import IdentityDataset
from Models.models import KochBackbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

print("Loading dataset...")

dataset = IdentityDataset(
    root_dir=root_dir,
    transform=transform
)

# -------------------------------------------------
# Select 25 identities
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
    idxs = identity_to_indices[name]

    # use up to 10 images per identity
    selected_indices.extend(idxs[:10])

subset = Subset(dataset, selected_indices)

loader = DataLoader(
    subset,
    batch_size=64,
    shuffle=False
)

print("Selected identities:", len(selected_names))
print("Selected images:", len(selected_indices))

# -------------------------------------------------
# Load best model
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

        embeddings = model(images)

        all_embeddings.append(embeddings.cpu())
        all_labels.extend(labels.numpy())

embeddings = torch.cat(all_embeddings).numpy()
labels = np.array(all_labels)

print("Embedding shape:", embeddings.shape)

# -------------------------------------------------
# t-SNE
# -------------------------------------------------

print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca"
)

proj = tsne.fit_transform(embeddings)

# -------------------------------------------------
# Plot
# -------------------------------------------------

print("Creating plot...")

plt.figure(figsize=(12, 10))

unique_labels = np.unique(labels)

for label in unique_labels:
    idxs = labels == label

    plt.scatter(
        proj[idxs, 0],
        proj[idxs, 1],
        s=30,
        alpha=0.8
    )

plt.title("t-SNE of Test Embeddings (Best Model)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)

output_path = "outputs/tsne_best_model.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved t-SNE plot to: {output_path}")