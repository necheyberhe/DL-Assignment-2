# DL-Assignment-2

## Students

| Name              | ID        |
| ----------------- | --------- |
| Noga Klein        | 326364007 |
| Nechi Berhe Weldu | 850164070 |

---

# Deep Metric Learning for Face Verification and One-Shot Recognition

PyTorch implementation and analysis of Siamese and deep metric-learning approaches for face verification and one-shot recognition on the LFW-a dataset.

The project investigates:

* Siamese BCE verification,
* contrastive learning,
* triplet metric learning,
* backbone architecture comparison,
* and pretrained transfer-learning baselines.

---

# Implemented Experiments

## Experiment 1 — Loss Function Comparison

Comparison of multiple metric-learning objectives using the same Siamese backbone:

* BCE Siamese baseline (Koch et al.)
* Contrastive loss
* Triplet loss with random mining
* Triplet loss with semi-hard negative mining

Evaluation includes:

* verification accuracy,
* ROC/AUC,
* one-shot classification,
* convergence analysis,
* and training-time comparison.

---

## Experiment 2 — Backbone Comparison

Comparison of backbone architectures under the same metric-learning objective:

* Modified Koch-style CNN
* ResNet-18 trained from scratch

The experiment analyzes:

* generalization,
* computational complexity,
* convergence behavior,
* and embedding quality.

---

## Experiment 3 — Frozen Pretrained Baseline

Transfer-learning baseline using:

* frozen ImageNet-pretrained ResNet-18,
* cosine similarity verification,
* no fine-tuning.

This experiment evaluates the transferability of generic pretrained visual representations to face verification tasks.

---

# Evaluation Protocol

All experiments were evaluated using:

* verification accuracy,
* ROC curves and AUC,
* N-way one-shot accuracy (N = 2, 5, 20),
* training and validation loss curves,
* convergence wall-clock time,
* t-SNE embedding visualization,
* intra/inter-class distance analysis,
* and qualitative failure-case analysis.

The same one-shot episode set was reused across all experiments for fair comparison.

---

# Framework

* PyTorch
* torchvision
* scikit-learn
* pandas
* matplotlib

---

# Dataset

## LFW-a (Aligned Labeled Faces in the Wild)

The project uses:

* aligned LFW-a face crops,
* standard verification pair splits.

Expected structure:

```text
Data/
├── lfw2/
│   └── lfw2/
├── pairsDevTrain.txt
└── pairsDevTest.txt
```

Update dataset paths inside scripts if local paths differ.

---

# Installation

Install dependencies:

```bash
pip install torch torchvision pandas matplotlib scikit-learn pillow thop
```

---

# How to Run

## Experiment 1 — Loss Function Comparison

### BCE Siamese baseline

```bash
python train_exp1.py --seed 42
```

### Contrastive loss models

```bash
python train_contrastive.py --seed 42 --margin 0.5
python train_contrastive.py --seed 42 --margin 1.0
python train_contrastive.py --seed 42 --margin 1.5
```

### Triplet-loss models

```bash
python train_triplet.py --seed 42
```

### Experiment 1 evaluation

```bash
python eval_exp1.py
python eval_oneshot_exp1.py
python plot_exp1_curves.py
python plot_exp1_training_time.py
```

---

## Experiment 2 — Backbone Comparison

### Train Koch CNN

```bash
python train_backbone_exp2.py --seed 42 --only_backbone koch
```

### Train ResNet18 from scratch

```bash
python train_backbone_exp2.py --seed 42 --only_backbone resnet18
```

### Experiment 2 evaluation

```bash
python eval_exp2.py
python eval_oneshot_exp2.py
python exp2_oneshot_barchart.py
python plot_exp2_curves.py
python plot_exp2_training_time.py
```

---

## Experiment 3 — Frozen Pretrained Baseline

```bash
python eval_exp3_pretrained.py
```

---

# Final Summary Table

Generate the combined results table:

```bash
python build_final_table.py
```

Output:

```text
outputs/final_summary_table.csv
```

---

# Qualitative Analysis

Run embedding-space visualization and qualitative analysis:

```bash
python distance_analysis.py
python tsne_analysis2.py
python failure_analysis.py
```

Outputs include:

* t-SNE visualization,
* distance-distribution histograms,
* false accepts,
* false rejects.

---

# Outputs

Generated plots and CSV files are saved in:

```text
outputs/
```

Model checkpoints are saved in:

```text
checkpoints/
```

---

# Main Findings

* Metric-learning objectives substantially outperformed BCE Siamese verification.
* Triplet semi-hard mining achieved the strongest embedding discrimination.
* The modified Koch CNN generalized better than ResNet18 trained from scratch.
* Frozen pretrained ImageNet features provided a surprisingly strong transfer-learning baseline despite requiring no task-specific optimization.
