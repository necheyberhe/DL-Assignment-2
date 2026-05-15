from pathlib import Path
import pandas as pd


output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# -------------------------
# Load result files
# -------------------------

exp1 = pd.read_csv("results_exp1_final_seed_42.csv")
exp2 = pd.read_csv("results_exp2_seed_42.csv")
exp3 = pd.read_csv("outputs/exp3_pretrained.csv")

exp1_auc = pd.read_csv("outputs/exp1_auc.csv")
exp2_auc = pd.read_csv("outputs/exp2_auc.csv")
exp1_one = pd.read_csv("outputs/exp1_oneshot_seed_42.csv")
exp2_one = pd.read_csv("outputs/exp2_oneshot_seed_42.csv")


# -------------------------
# Experiment 1
# -------------------------

exp1_table = exp1.copy()

exp1_table = exp1_table.merge(
    exp1_auc,
    on="method",
    how="left"
)

exp1_table = exp1_table.merge(
    exp1_one,
    on="method",
    how="left"
)

exp1_table["model"] = exp1_table["method"]
exp1_table["experiment"] = "Exp1"


# -------------------------
# Experiment 2
# -------------------------

exp2_table = exp2.copy()

name_map = {
    "koch": "KochCNN",
    "resnet18": "ResNet18Scratch",
}

exp2_table["model"] = exp2_table["backbone"].map(name_map)

exp2_table = exp2_table.merge(
    exp2_auc.rename(columns={"backbone": "model"}),
    on="model",
    how="left"
)

exp2_table = exp2_table.merge(
    exp2_one.rename(columns={"backbone": "model"}),
    on="model",
    how="left"
)

exp2_table["experiment"] = "Exp2"


# -------------------------
# Experiment 3
# -------------------------

exp3_table = pd.DataFrame([{
    "experiment": "Exp3",
    "model": "Frozen ImageNet ResNet18",
    "loss": "none",
    "margin": "",
    "seed": 42,
    "val_threshold": exp3.loc[0, "threshold"],
    "test_accuracy": exp3.loc[0, "test_accuracy"],
    "auc": exp3.loc[0, "auc"],
    "oneshot_acc_N2": exp3.loc[0, "oneshot_N2"],
    "oneshot_acc_N5": exp3.loc[0, "oneshot_N5"],
    "oneshot_acc_N20": exp3.loc[0, "oneshot_N20"],
    "params": "",
    "macs": "",
    "approx_flops": "",
    "train_time_sec": 0,
    "best_epoch": "",
    "checkpoint": "Frozen pretrained ImageNet ResNet18",
}])

# -------------------------
# Combine
# -------------------------

keep_cols = [
    "experiment",
    "model",
    "loss",
    "margin",
    "seed",
    "val_threshold",
    "val_accuracy",
    "test_accuracy",
    "auc",
    "oneshot_acc_N2",
    "oneshot_acc_N5",
    "oneshot_acc_N20",
    "params",
    "macs",
    "approx_flops",
    "train_time_sec",
    "best_epoch",
    "checkpoint",
]

for df in [exp1_table, exp2_table, exp3_table]:
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""

final = pd.concat(
    [
        exp1_table[keep_cols],
        exp2_table[keep_cols],
        exp3_table[keep_cols],
    ],
    ignore_index=True
)

# Add training time in minutes
final["train_time_min"] = pd.to_numeric(
    final["train_time_sec"],
    errors="coerce"
) / 60.0

# Round numeric columns
for col in [
    "val_threshold",
    "val_accuracy",
    "test_accuracy",
    "auc",
    "oneshot_acc_N2",
    "oneshot_acc_N5",
    "oneshot_acc_N20",
    "train_time_min",
]:
    final[col] = pd.to_numeric(final[col], errors="coerce").round(4)

# Save
out_path = output_dir / "final_summary_table.csv"
final.to_csv(out_path, index=False)

print(final)
print(f"\nSaved final summary table to: {out_path}")