import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("outputs/exp2_oneshot_seed_42.csv")
# Categories
tasks = ["N2", "N5", "N20"]

koch = [
    df.loc[df["backbone"] == "KochCNN", "oneshot_acc_N2"].values[0],
    df.loc[df["backbone"] == "KochCNN", "oneshot_acc_N5"].values[0],
    df.loc[df["backbone"] == "KochCNN", "oneshot_acc_N20"].values[0],
]

resnet = [
    df.loc[df["backbone"] == "ResNet18Scratch", "oneshot_acc_N2"].values[0],
    df.loc[df["backbone"] == "ResNet18Scratch", "oneshot_acc_N5"].values[0],
    df.loc[df["backbone"] == "ResNet18Scratch", "oneshot_acc_N20"].values[0],
]

x = np.arange(len(tasks))
width = 0.35

plt.figure(figsize=(7,5))

plt.bar(x - width/2, koch, width, label="Koch CNN")
plt.bar(x + width/2, resnet, width, label="ResNet-18")

plt.xticks(x, tasks)
plt.ylabel("One-shot Accuracy")
plt.xlabel("Task Difficulty")
plt.title("Experiment 2 One-shot Accuracy")
plt.ylim(0, 1.0)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
df = pd.read_csv("outputs/exp2_oneshot_seed_42.csv")
plt.savefig("outputs/exp2_oneshot_bar.png", dpi=400)
plt.close()

print("Saved to outputs/exp2_oneshot_bar.png")