import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_exp1_final_seed_42.csv")

methods = df["method"]
times_min = df["train_time_sec"] / 60.0

plt.figure(figsize=(7,5))

plt.bar(methods, times_min)

plt.ylabel("Training Time (minutes)")
plt.xlabel("Method")
plt.title("Experiment 1 Convergence Time")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "outputs/exp1_training_time.png",
    dpi=400
)

plt.close()

print("Saved outputs/exp1_training_time.png")