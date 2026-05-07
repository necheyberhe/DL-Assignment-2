import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from sklearn.manifold import TSNE
from torchvision import transforms
from Data.datasets import IdentityDataset
from Models.models import KochBackbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"

transform = transforms.Compose([
    transforms.Resize((105,105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = IdentityDataset(root_dir, transform=transform)

# Sample identities
identities = list(dataset.name_to_label.keys())
selected = random.sample(identities, 25)

filtered = [
    (img,label,name)
    for img,label,name in dataset.samples
    if name in selected
]

images = torch.stack([x[0] for x in filtered]).to(device)
labels = [x[2] for x in filtered]

# Load best model (replace if needed)
model = KochBackbone(128).to(device)
ckpt = torch.load("checkpoints/exp2_best_koch.pt", map_location=device)
state = {k:v for k,v in ckpt["backbone_state_dict"].items()
         if "total_ops" not in k}
model.load_state_dict(state, strict=False)
model.eval()

with torch.no_grad():
    emb = model(images).cpu().numpy()

tsne = TSNE(n_components=2, perplexity=30)
proj = tsne.fit_transform(emb)

plt.figure(figsize=(8,6))

for i, name in enumerate(set(labels)):
    idx = [j for j,l in enumerate(labels) if l==name]
    plt.scatter(proj[idx,0], proj[idx,1], label=name)

plt.legend(fontsize=6)
plt.title("t-SNE Embedding (Best Model)")
plt.savefig("outputs/tsne_plot.png", dpi=300)
plt.show()