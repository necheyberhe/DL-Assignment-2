import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_exp2_seed_42.csv")

names = df["backbone"]
times = df["train_time_sec"] / 60.0

plt.figure(figsize=(7,5))

plt.bar(names, times)

plt.xlabel("Backbone")
plt.ylabel("Training Time (minutes)")
plt.title("Experiment 2 Convergence Time")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "outputs/exp2_training_time.png",
    dpi=400
)

plt.close()

print("Saved outputs/exp2_training_time.png")