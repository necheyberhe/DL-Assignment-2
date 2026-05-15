# import pandas as pd
# import matplotlib.pyplot as plt


# # =========================
# # FILES
# # =========================

# files = {
#     "Contrastive": "outputs/history_contrastive_seed_42.csv",
#     "KochCNN": "outputs/history_exp2_koch_seed_42.csv",
#     "ResNet18": "outputs/history_exp2_resnet18_seed_42.csv",
# }


# # =========================
# # TRAINING LOSS
# # =========================

# plt.figure(figsize=(8, 6))

# for name, path in files.items():

#     df = pd.read_csv(path)

#     plt.plot(
#         df["epoch"],
#         df["train_loss"],
#         linewidth=2,
#         label=name
#     )

# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
# plt.title("Training Loss Curves")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(
#     "outputs/training_loss_curves.png",
#     dpi=400
# )

# plt.close()


# # =========================
# # VALIDATION ACCURACY
# # =========================

# plt.figure(figsize=(8, 6))

# for name, path in files.items():

#     df = pd.read_csv(path)

#     plt.plot(
#         df["epoch"],
#         df["val_best_acc"],
#         linewidth=2,
#         label=name
#     )

# plt.xlabel("Epoch")
# plt.ylabel("Validation Accuracy")
# plt.title("Validation Accuracy Curves")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(
#     "outputs/validation_accuracy_curves.png",
#     dpi=400
# )

# plt.close()

# print("Saved:")
# print("outputs/training_loss_curves.png")
# print("outputs/validation_accuracy_curves.png")

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SEED = 42
results_csv = Path("results_exp1_final_seed_42.csv")
# results_csv = Path(f"results_exp1_seed_{SEED}.csv")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

results = pd.read_csv(results_csv)

required_cols = {"method", "history_csv"}
missing = required_cols - set(results.columns)
if missing:
    raise ValueError(f"Missing columns in {results_csv}: {missing}")

# If multiple margins exist, keep the selected/best row per method
# because results_exp1_seed_42.csv should contain only selected rows.
rows = results.dropna(subset=["history_csv"])


def load_histories(rows):
    histories = {}

    for _, row in rows.iterrows():
        method = row["method"]
        history_path = Path(row["history_csv"])

        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file for {method}: {history_path}")

        hist = pd.read_csv(history_path)
        histories[method] = hist

    return histories


def plot_metric(histories, metric, ylabel, title, output_name):
    plt.figure(figsize=(8, 6))

    for method, hist in histories.items():
        if metric not in hist.columns:
            print(f"Skipping {method}: no column '{metric}'")
            continue

        y = pd.to_numeric(hist[metric], errors="coerce")

        if y.notna().sum() == 0:
            print(f"Skipping {method}: column '{metric}' is empty/non-numeric")
            continue

        plt.plot(
            hist["epoch"],
            y,
            linewidth=2,
            label=method
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = output_dir / output_name
    plt.savefig(out_path, dpi=400)
    plt.close()

    print(f"Saved: {out_path}")


histories = load_histories(rows)

plot_metric(
    histories,
    metric="train_loss",
    ylabel="Training Loss",
    title="Experiment 1 Training Loss Curves",
    output_name=f"exp1_training_loss_curves_seed_{SEED}.png",
)

plot_metric(
    histories,
    metric="val_loss",
    ylabel="Validation Loss",
    title="Experiment 1 Validation Loss Curves",
    output_name=f"exp1_validation_loss_curves_seed_{SEED}.png",
)

plot_metric(
    histories,
    metric="val_best_acc",
    ylabel="Validation Accuracy",
    title="Experiment 1 Validation Accuracy Curves",
    output_name=f"exp1_validation_accuracy_curves_seed_{SEED}.png",
)