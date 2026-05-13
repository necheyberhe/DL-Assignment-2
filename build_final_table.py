import pandas as pd


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Existing columns: {df.columns.tolist()}")


def normalize_model_name(x):
    mapping = {
        "BCE_L1": "BCE_L1",
        "Contrastive": "Contrastive",
        "Triplet_Random": "Triplet_Random",
        "Triplet_Semihard": "Triplet_Semihard",

        "koch": "KochCNN_Exp2",
        "KochCNN": "KochCNN",
        "KochCNN_Exp2": "KochCNN_Exp2",

        "resnet18": "ResNet18_Scratch",
        "ResNet18Scratch": "ResNet18_Scratch",
        "ResNet18_Scratch": "ResNet18_Scratch",

        "ResNet18_Pretrained": "ResNet18_Pretrained_Frozen",
        "ResNet18_Pretrained_Frozen": "ResNet18_Pretrained_Frozen",
    }
    return mapping.get(x, x)


# =========================
# LOAD FILES
# =========================

exp1 = pd.read_csv("results_exp1.csv", header=None)

base_cols = [
    "method",
    "backbone",
    "loss",
    "margin",
    "selection_metric",
    "best_epoch",
    "val_threshold",
    "val_accuracy",
    "test_accuracy",
    "checkpoint",
    "seed",
    "train_time_sec",
    "max_epochs",
    "patience",
    "min_delta",
    "stopping_rule",
]

exp1.columns = base_cols[:len(exp1.columns)]
exp2 = pd.read_csv("results_exp2.csv")
exp3 = pd.read_csv("outputs/exp3_pretrained.csv")

auc = pd.read_csv("outputs/all_models_auc.csv")
oneshot1 = pd.read_csv("outputs/exp1_oneshot.csv")
oneshot2 = pd.read_csv("outputs/exp2_oneshot.csv")


# =========================
# EXP1
# =========================

exp1_model_col = find_col(exp1, ["model", "method"])
exp1_test_col = find_col(exp1, ["test_accuracy", "test_acc"])

exp1_clean = exp1.copy()
exp1_clean["model"] = exp1_clean[exp1_model_col]
exp1_clean["test_accuracy"] = exp1_clean[exp1_test_col]

if "backbone" not in exp1_clean.columns:
    exp1_clean["backbone"] = "KochCNN"

if "loss" not in exp1_clean.columns:
    exp1_clean["loss"] = exp1_clean["model"]

exp1_clean["params"] = 10777024
exp1_clean["approx_flops"] = 1996816384.0

exp1_clean = exp1_clean[
    ["model", "backbone", "loss", "test_accuracy", "params", "approx_flops"]
]


# =========================
# EXP2
# =========================

exp2_model_col = find_col(exp2, ["model", "backbone"])
exp2_test_col = find_col(exp2, ["test_accuracy", "test_acc", "test_acc_at_val_threshold"])

exp2_clean = exp2.copy()
exp2_clean["model"] = exp2_clean[exp2_model_col]
exp2_clean["test_accuracy"] = exp2_clean[exp2_test_col]
exp2_clean["loss"] = "triplet_semihard"

exp2_clean = exp2_clean.rename(columns={
    "approx_flops": "approx_flops",
    "params": "params",
})

exp2_clean = exp2_clean[
    ["model", "backbone", "loss", "test_accuracy", "params", "approx_flops", "train_time_sec"]
]


# =========================
# EXP3
# =========================

exp3_test_col = find_col(exp3, ["test_accuracy", "test_acc"])

exp3_clean = pd.DataFrame([{
    "model": "ResNet18_Pretrained_Frozen",
    "backbone": "ResNet18",
    "loss": "frozen_pretrained_cosine",
    "test_accuracy": exp3[exp3_test_col].iloc[0],
    "params": 11689512,
    "approx_flops": 952234880.0,
    "train_time_sec": 0.0,
}])


# =========================
# COMBINE CORE
# =========================

df = pd.concat([exp1_clean, exp2_clean, exp3_clean], ignore_index=True)

df["model"] = df["model"].apply(normalize_model_name)
df["backbone"] = df["backbone"].apply(normalize_model_name)

exp1_models = [
    "BCE_L1",
    "Contrastive",
    "Triplet_Random",
    "Triplet_Semihard",
]

df.loc[df["model"].isin(exp1_models), "backbone"] = "KochCNN"
# =========================
# AUC
# =========================

if "model" not in auc.columns:
    auc = auc.rename(columns={"backbone": "model", "method": "model"})

auc["model"] = auc["model"].apply(normalize_model_name)

df = df.merge(
    auc[["model", "auc"]],
    on="model",
    how="left"
)


# =========================
# ONE-SHOT
# =========================

if "method" in oneshot1.columns:
    oneshot1 = oneshot1.rename(columns={"method": "model"})

if "backbone" in oneshot2.columns:
    oneshot2 = oneshot2.rename(columns={"backbone": "model"})

exp3_oneshot = pd.DataFrame([{
    "model": "ResNet18_Pretrained_Frozen",
    "oneshot_acc_N2": exp3["oneshot_N2"].iloc[0],
    "oneshot_acc_N5": exp3["oneshot_N5"].iloc[0],
    "oneshot_acc_N20": exp3["oneshot_N20"].iloc[0],
}])

oneshot = pd.concat([oneshot1, oneshot2, exp3_oneshot], ignore_index=True)
oneshot["model"] = oneshot["model"].apply(normalize_model_name)

df = df.merge(
    oneshot[["model", "oneshot_acc_N2", "oneshot_acc_N5", "oneshot_acc_N20"]],
    on="model",
    how="left"
)


# =========================
# CLEAN / SAVE
# =========================

df = df.drop_duplicates(subset=["model"], keep="last")

df = df[[
    "model",
    "backbone",
    "loss",
    "test_accuracy",
    "auc",
    "oneshot_acc_N2",
    "oneshot_acc_N5",
    "oneshot_acc_N20",
    "params",
    "approx_flops",
    "train_time_sec",
]]

df.to_csv("outputs/final_results.csv", index=False)

print("\nFinal table:")
print(df)