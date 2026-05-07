# DL-Assignment-2

# Deep Metric Learning for Face Verification and One-Shot Recognition

PyTorch implementation and analysis of Siamese and metric-learning approaches on the LFW-a dataset.

## Implemented Experiments

### Experiment 1 — Loss Function Comparison
- BCE Siamese baseline (Koch et al.)
- Contrastive loss
- Triplet loss with semi-hard negative mining
- Random triplet ablation

### Experiment 2 — Backbone Comparison
- Modified Koch-style CNN
- ResNet-18 trained from scratch
- Parameter-matched comparison

### Experiment 3 — Frozen Pretrained Baseline
- Frozen ImageNet-pretrained ResNet-18
- Cosine similarity verification

## Evaluation
- Verification accuracy
- ROC / AUC
- N-way one-shot evaluation (N = 2, 5, 20)
- t-SNE embedding visualization
- Intra/inter-class distance analysis

## Framework
- PyTorch

## Dataset
- LFW-a (aligned LFW)
