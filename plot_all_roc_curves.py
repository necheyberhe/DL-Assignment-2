import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Load AUC tables
# -------------------------

exp1_auc = pd.read_csv("outputs/exp1_auc.csv")
exp2_auc = pd.read_csv("outputs/exp2_auc.csv")
exp3 = pd.read_csv("outputs/exp3_pretrained.csv")

# -------------------------
# Placeholder ROC loading
# -------------------------
# Assumes ROC data CSVs exist.
# If not, regenerate ROC points from evaluation scripts.

# Example:
# outputs/roc_bce.csv
# outputs/roc_triplet.csv
# etc.

roc_files = {
    "Triplet_Random": "outputs/roc_exp1_triplet_random.csv",
    "Triplet_Semihard": "outputs/roc_exp1_triplet_semihard.csv",

    "KochCNN": "outputs/roc_exp2_kochcnn.csv",
    "ResNet18Scratch": "outputs/roc_exp2_resnet18scratch.csv",

    "Frozen ImageNet ResNet18": "outputs/roc_exp3_frozen_resnet18.csv",
}

# -------------------------
# AUC lookup
# -------------------------

auc_lookup = {}

for _, row in exp1_auc.iterrows():
    auc_lookup[row["method"]] = row["auc"]

for _, row in exp2_auc.iterrows():
    auc_lookup[row["backbone"]] = row["auc"]

auc_lookup["Frozen ImageNet ResNet18"] = exp3.loc[0, "auc"]

# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(10, 8))

for model_name, roc_path in roc_files.items():

    roc_df = pd.read_csv(roc_path)

    plt.plot(
        roc_df["fpr"],
        roc_df["tpr"],
        linewidth=2,
        label=f"{model_name} (AUC={auc_lookup[model_name]:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves Across All Experiments")

plt.legend(fontsize=9)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/all_roc_curves.png",
    dpi=400
)

plt.close()

print("Saved: outputs/all_roc_curves.png")