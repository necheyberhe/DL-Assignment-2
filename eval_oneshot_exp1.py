import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from Data.datasets import PairDataset, identities_from_pairs
from Models.models import KochBackbone, SiameseBCEHead


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = Path(r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\lfw2\lfw2")
pairs_test = r"D:\Masters Study\2ndyear\Deep_Learning\DL-Assignment-2\Data\pairsDevTest.txt"
results_csv = "results_exp1.csv"

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

episodes_path = output_dir / "oneshot_episodes.json"

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def get_image_path(name, idx):
    return root_dir / name / f"{name}_{int(idx):04d}.jpg"


def collect_test_identity_images():
    test_pairs = PairDataset(pairs_test, root_dir, transform=None)
    test_names = identities_from_pairs(test_pairs.pairs)

    identity_to_images = {}

    for name in test_names:
        person_dir = root_dir / name
        images = sorted(person_dir.glob("*.jpg"))

        if len(images) >= 2:
            identity_to_images[name] = [str(p) for p in images]

    return identity_to_images


def create_episodes(identity_to_images, n_values=(2, 5, 20), episodes_per_n=200, seed=42):
    rng = random.Random(seed)
    episodes = {}

    valid_ids = list(identity_to_images.keys())

    for n in n_values:
        n_episodes = []

        eligible_query_ids = [
            name for name, imgs in identity_to_images.items()
            if len(imgs) >= 2
        ]

        for _ in range(episodes_per_n):
            query_id = rng.choice(eligible_query_ids)

            query_img, positive_img = rng.sample(
                identity_to_images[query_id],
                2
            )

            distractor_ids = rng.sample(
                [x for x in valid_ids if x != query_id],
                n - 1
            )

            support = [
                {
                    "identity": query_id,
                    "image": positive_img,
                    "correct": True
                }
            ]

            for did in distractor_ids:
                support.append({
                    "identity": did,
                    "image": rng.choice(identity_to_images[did]),
                    "correct": False
                })

            rng.shuffle(support)

            correct_index = [
                i for i, item in enumerate(support)
                if item["correct"]
            ][0]

            n_episodes.append({
                "query_identity": query_id,
                "query_image": query_img,
                "support": support,
                "correct_index": correct_index
            })

        episodes[str(n)] = n_episodes

    return episodes


def load_or_create_episodes():
    if episodes_path.exists():
        with open(episodes_path, "r") as f:
            return json.load(f)

    identity_to_images = collect_test_identity_images()
    episodes = create_episodes(identity_to_images)

    with open(episodes_path, "w") as f:
        json.dump(episodes, f, indent=2)

    return episodes


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)


def load_bce_checkpoint(path):
    checkpoint = torch.load(path, map_location=device)

    backbone = KochBackbone(embedding_dim=128).to(device)
    head = SiameseBCEHead(embedding_dim=128).to(device)

    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])

    backbone.eval()
    head.eval()

    return backbone, head


def load_backbone_checkpoint(path):
    checkpoint = torch.load(path, map_location=device)

    backbone = KochBackbone(embedding_dim=128).to(device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()

    return backbone


@torch.no_grad()
def score_episode_bce(backbone, head, episode):
    query = load_image(episode["query_image"]).unsqueeze(0).to(device)

    support_imgs = [
        load_image(item["image"])
        for item in episode["support"]
    ]

    support = torch.stack(support_imgs).to(device)

    query_batch = query.repeat(support.size(0), 1, 1, 1)

    zq = backbone(query_batch)
    zs = backbone(support)

    logits = head(zq, zs)
    scores = torch.sigmoid(logits)

    return torch.argmax(scores).item()


@torch.no_grad()
def score_episode_distance(backbone, episode):
    query = load_image(episode["query_image"]).unsqueeze(0).to(device)

    support_imgs = [
        load_image(item["image"])
        for item in episode["support"]
    ]

    support = torch.stack(support_imgs).to(device)

    query_batch = query.repeat(support.size(0), 1, 1, 1)

    zq = backbone(query_batch)
    zs = backbone(support)

    distances = F.pairwise_distance(zq, zs, p=2)

    # nearest support image wins
    return torch.argmin(distances).item()


def evaluate_model_oneshot(method, checkpoint_path, episodes):
    results = {}

    if method == "BCE_L1":
        backbone, head = load_bce_checkpoint(checkpoint_path)
        scorer = lambda ep: score_episode_bce(backbone, head, ep)
    else:
        backbone = load_backbone_checkpoint(checkpoint_path)
        scorer = lambda ep: score_episode_distance(backbone, ep)

    for n, eps in episodes.items():
        correct = 0

        for ep in eps:
            pred_idx = scorer(ep)

            if pred_idx == ep["correct_index"]:
                correct += 1

        acc = correct / len(eps)
        results[int(n)] = acc

    return results


def main():
    episodes = load_or_create_episodes()

    print(f"Loaded one-shot episodes from: {episodes_path}")
    for n, eps in episodes.items():
        print(f"N={n}: {len(eps)} episodes")

    results_df = pd.read_csv(results_csv)

    rows = []

    for _, row in results_df.iterrows():
        method = row["method"]
        checkpoint_path = row["checkpoint"]

        print(f"\nEvaluating {method}")

        accs = evaluate_model_oneshot(
            method=method,
            checkpoint_path=checkpoint_path,
            episodes=episodes
        )

        rows.append({
            "method": method,
            "oneshot_acc_N2": accs[2],
            "oneshot_acc_N5": accs[5],
            "oneshot_acc_N20": accs[20],
        })

        print(accs)

    out_df = pd.DataFrame(rows)
    out_path = output_dir / "exp1_oneshot.csv"
    out_df.to_csv(out_path, index=False)

    print("\nOne-shot results:")
    print(out_df)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()