import pandas as pd
import matplotlib.pyplot as plt

files = {
    "KochCNN": "outputs/history_exp2_koch_seed_42.csv",
    "ResNet18": "outputs/history_exp2_resnet18_seed_42.csv",
}

# -------------------------
# TRAINING LOSS
# -------------------------

plt.figure(figsize=(8,6))

for name, path in files.items():

    df = pd.read_csv(path)

    plt.plot(
        df["epoch"],
        df["train_loss"],
        linewidth=2,
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Experiment 2 Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/exp2_training_loss.png",
    dpi=400
)

plt.close()

# -------------------------
# VALIDATION LOSS
# -------------------------

plt.figure(figsize=(8,6))

for name, path in files.items():

    df = pd.read_csv(path)

    plt.plot(
        df["epoch"],
        df["val_loss"],
        linewidth=2,
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Experiment 2 Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/exp2_validation_loss.png",
    dpi=400
)

plt.close()

# -------------------------
# VALIDATION ACCURACY
# -------------------------

plt.figure(figsize=(8,6))

for name, path in files.items():

    df = pd.read_csv(path)

    plt.plot(
        df["epoch"],
        df["val_best_acc"],
        linewidth=2,
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Experiment 2 Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/exp2_validation_accuracy.png",
    dpi=400
)

plt.close()

print("Saved Experiment 2 plots.")