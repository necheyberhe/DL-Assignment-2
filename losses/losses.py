import torch
import torch.nn.functional as F


def bce_l1_loss(logits, labels):
    labels = labels.float()
    return F.binary_cross_entropy_with_logits(logits, labels)


def contrastive_loss(z1, z2, labels, margin=1.0):
    d = F.pairwise_distance(z1, z2, p=2)

    positive = labels * d.pow(2)
    negative = (1.0 - labels) * F.relu(margin - d).pow(2)

    return (positive + negative).mean()

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)

    return F.relu(d_ap - d_an + margin).mean()

def semihard_triplets(embeddings, labels, margin=0.2):
    device = embeddings.device
    labels = labels.view(-1)

    dist = torch.cdist(embeddings, embeddings, p=2)

    anchors = []
    positives = []
    negatives = []

    batch_size = embeddings.size(0)

    for i in range(batch_size):
        same = labels == labels[i]
        diff = labels != labels[i]

        same[i] = False

        pos_indices = torch.where(same)[0]
        neg_indices = torch.where(diff)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        for p_idx in pos_indices:
            d_ap = dist[i, p_idx]

            semi_hard_mask = (dist[i, neg_indices] > d_ap) & (
                dist[i, neg_indices] < d_ap + margin
            )

            semi_hard_negs = neg_indices[semi_hard_mask]

            if len(semi_hard_negs) > 0:
                rand_idx = torch.randint(
                    low=0,
                    high=len(semi_hard_negs),
                    size=(1,),
                    device=device
                ).item()

                n_idx = semi_hard_negs[rand_idx]

                anchors.append(int(i))
                positives.append(int(p_idx.item()))
                negatives.append(int(n_idx.item()))

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


# z = torch.randn(8, 128)
# labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

# triplets = semihard_triplets(z, labels, margin=10.0)

# if triplets is None:
#     print("No triplets found")
# else:
#     a, p, n = triplets
#     print(a.shape, p.shape, n.shape)
#     print(triplet_loss(a, p, n))