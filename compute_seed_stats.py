import pandas as pd
import numpy as np

files = [
    "results_exp2_seed_42.csv",
    "results_exp2_seed_123.csv",
    "results_exp2_seed_999.csv",
]

accs = []

for f in files:
    df = pd.read_csv(f)

    row = df[df["backbone"] == "koch"]

    acc = row["test_accuracy"].iloc[0]
    accs.append(acc)

mean = np.mean(accs)
std = np.std(accs)

print("Koch Exp2 test accuracies:", accs)
print(f"Mean ± std = {mean:.4f} ± {std:.4f}")