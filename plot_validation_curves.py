import pandas as pd
import matplotlib.pyplot as plt


files = {
    "Contrastive": "outputs/history_contrastive_seed_42.csv",
    "KochCNN": "outputs/history_exp2_koch_seed_42.csv",
    "ResNet18": "outputs/history_exp2_resnet18_seed_42.csv",
}


plt.figure(figsize=(8, 6))

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
plt.title("Validation Accuracy Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "outputs/validation_accuracy_curves.png",
    dpi=400
)

plt.close()

print("Saved: outputs/validation_accuracy_curves.png")