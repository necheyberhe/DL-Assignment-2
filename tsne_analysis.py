import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torchvision import transforms

from Models.models import KochBackbone


# =========================
# SETTINGS
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = Path(r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2")

checkpoint = "checkpoints/exp2_best_koch.pt"

NUM_IDENTITIES = 25
SEED = 42

random.seed(SEED)


# =========================
# MODEL
# =========================

model = KochBackbone(embedding_dim=128).to(device)

ckpt = torch.load(checkpoint, map_location=device)

state = ckpt["backbone_state_dict"]

state = {
    k: v for k, v in state.items()
    if "total_ops" not in k and "total_params" not in k
}

model.load_state_dict(state, strict=False)

model.eval()


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# =========================
# SAMPLE IDENTITIES
# =========================

all_ids = [
    d for d in root_dir.iterdir()
    if d.is_dir()
]

sampled_ids = random.sample(all_ids, NUM_IDENTITIES)

embeddings = []
labels = []


# =========================
# EXTRACT EMBEDDINGS
# =========================

with torch.no_grad():

    for identity_dir in sampled_ids:

        identity_name = identity_dir.name

        for img_path in identity_dir.glob("*.jpg"):

            img = Image.open(img_path).convert("RGB")

            x = transform(img).unsqueeze(0).to(device)

            z = model(x)

            embeddings.append(
                z.squeeze(0).cpu().numpy()
            )

            labels.append(identity_name)


embeddings = np.array(embeddings)


# =========================
# TSNE
# =========================

tsne = TSNE(
    n_components=2,
    perplexity=15,
    random_state=SEED
)

proj = tsne.fit_transform(embeddings)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 8))

unique_labels = sorted(set(labels))

for label in unique_labels:

    idx = [i for i, x in enumerate(labels) if x == label]

    plt.scatter(
        proj[idx, 0],
        proj[idx, 1],
        s=20,
        alpha=0.8,
        label=label
    )

plt.title("t-SNE of Test Embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# optional:
# plt.legend(fontsize=6)

plt.tight_layout()

plt.savefig(
    "outputs/tsne_embeddings.png",
    dpi=400
)

plt.close()

print("Saved: outputs/tsne_embeddings.png")
##############################################
# =========================
# DISTANCE ANALYSIS
# =========================

emb_tensor = torch.tensor(embeddings)

dist = torch.cdist(emb_tensor, emb_tensor)

intra = []
inter = []

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):

        d = dist[i, j].item()

        if labels[i] == labels[j]:
            intra.append(d)
        else:
            inter.append(d)

print("\nDistance analysis:")
print(f"Mean intra-class distance: {np.mean(intra):.4f}")
print(f"Std intra-class distance: {np.std(intra):.4f}")

print(f"Mean inter-class distance: {np.mean(inter):.4f}")
print(f"Std inter-class distance: {np.std(inter):.4f}")