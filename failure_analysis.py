import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader

from Data.datasets import PairDataset
from Models.models import KochBackbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2"
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"

threshold = 0.8907807469367981

transform = transforms.Compose([
    transforms.Resize((105,105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


dataset = PairDataset(
    pairs_test,
    root_dir,
    transform
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)


# -------------------------
# MODEL
# -------------------------

model = KochBackbone(128).to(device)

ckpt = torch.load(
   "checkpoints/exp2_best_koch_seed_42.pt",
    map_location=device
)

state = {
    k:v for k,v in ckpt["backbone_state_dict"].items()
    if "total_ops" not in k and "total_params" not in k
}

model.load_state_dict(state, strict=False)

model.eval()


false_accepts = []
false_rejects = []


# -------------------------
# EVAL
# -------------------------

with torch.no_grad():

    for img1, img2, label in loader:

        img1 = img1.to(device)
        img2 = img2.to(device)

        z1 = model(img1)
        z2 = model(img2)

        dist = F.pairwise_distance(z1, z2)

        pred_same = dist.item() < threshold

        true_same = bool(label.item())

        # False Accept
        if pred_same and not true_same:

            false_accepts.append(
                (img1.cpu(), img2.cpu(), dist.item())
            )

        # False Reject
        if not pred_same and true_same:

            false_rejects.append(
                (img1.cpu(), img2.cpu(), dist.item())
            )

        if len(false_accepts) >= 3 and len(false_rejects) >= 3:
            break


# -------------------------
# SAVE FIGURES
# -------------------------

def denorm(x):
    return x * 0.5 + 0.5


os.makedirs("outputs/failure_cases", exist_ok=True)


def save_cases(cases, title, filename):

    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(6, 3 * len(cases))
    )

    fig.suptitle(title)

    for i, (img1, img2, dist) in enumerate(cases):

        im1 = denorm(img1.squeeze()).permute(1,2,0).numpy()
        im2 = denorm(img2.squeeze()).permute(1,2,0).numpy()

        axes[i,0].imshow(im1)
        axes[i,0].set_title(f"Dist={dist:.3f}")
        axes[i,0].axis("off")

        axes[i,1].imshow(im2)
        axes[i,1].set_title(f"Dist={dist:.3f}")
        axes[i,1].axis("off")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_cases(
    false_accepts,
    "False Accepts",
    "outputs/failure_cases/false_accepts.png"
)

save_cases(
    false_rejects,
    "False Rejects",
    "outputs/failure_cases/false_rejects.png"
)

print("Saved failure-case figures.")