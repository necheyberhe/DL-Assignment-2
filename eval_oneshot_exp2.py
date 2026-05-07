import json
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms

from Models.models import KochBackbone
from Models.resnet import ResNet18Backbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

episodes_path = Path("outputs/oneshot_episodes.json")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

def load_exp2_backbone(backbone_name, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if backbone_name == "koch":
        model = KochBackbone(embedding_dim=128).to(device)
    elif backbone_name == "resnet18":
        model = ResNet18Backbone(embedding_dim=128).to(device)
    else:
        raise ValueError(backbone_name)

    state = checkpoint["backbone_state_dict"]

    clean_state = {
        k: v for k, v in state.items()
        if "total_ops" not in k and "total_params" not in k
    }

    model.load_state_dict(clean_state, strict=False)
    model.eval()

    return model


@torch.no_grad()
def score_episode(model, episode):
    query = load_image(episode["query_image"]).unsqueeze(0).to(device)

    support_imgs = [
        load_image(item["image"])
        for item in episode["support"]
    ]

    support = torch.stack(support_imgs).to(device)
    query_batch = query.repeat(support.size(0), 1, 1, 1)

    zq = model(query_batch)
    zs = model(support)

    distances = F.pairwise_distance(zq, zs, p=2)

    return torch.argmin(distances).item()


def evaluate_model(model, episodes):
    accs = {}

    for n, eps in episodes.items():
        correct = 0

        for ep in eps:
            pred_idx = score_episode(model, ep)

            if pred_idx == ep["correct_index"]:
                correct += 1

        accs[int(n)] = correct / len(eps)

    return accs


with open(episodes_path, "r") as f:
    episodes = json.load(f)

models = [
    ("KochCNN", "koch", "checkpoints/exp2_best_koch.pt"),
    ("ResNet18Scratch", "resnet18", "checkpoints/exp2_best_resnet18.pt"),
]

rows = []

for display_name, backbone_name, ckpt in models:
    print(f"Evaluating {display_name}")

    model = load_exp2_backbone(backbone_name, ckpt)
    accs = evaluate_model(model, episodes)

    rows.append({
        "backbone": display_name,
        "oneshot_acc_N2": accs[2],
        "oneshot_acc_N5": accs[5],
        "oneshot_acc_N20": accs[20],
        "checkpoint": ckpt,
    })

    print(accs)

df = pd.DataFrame(rows)
out_path = output_dir / "exp2_oneshot.csv"
df.to_csv(out_path, index=False)

print(df)
print(f"Saved to: {out_path}")