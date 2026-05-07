import pandas as pd
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Existing columns: {df.columns.tolist()}")


exp1 = pd.read_csv("results_exp1.csv")
exp2 = pd.read_csv("results_exp2.csv")

name_map = {
    "koch": "KochCNN",
    "resnet18": "ResNet18Scratch"
}

exp2["backbone"] = exp2["backbone"].replace(name_map)

exp3 = pd.read_csv("outputs/exp3_pretrained.csv")
exp3_oneshot = pd.DataFrame([{
    "model": "ResNet18_Pretrained",
    "oneshot_acc_N2": exp3["oneshot_N2"].iloc[0],
    "oneshot_acc_N5": exp3["oneshot_N5"].iloc[0],
    "oneshot_acc_N20": exp3["oneshot_N20"].iloc[0],
}])


auc = pd.read_csv("outputs/all_models_auc.csv")
oneshot1 = pd.read_csv("outputs/exp1_oneshot.csv")
oneshot2 = pd.read_csv("outputs/exp2_oneshot.csv")


# ----- Exp1 -----
exp1_model_col = find_col(exp1, ["model", "method"])
exp1_test_col = find_col(exp1, ["test_accuracy", "test_acc"])

exp1_clean = exp1[[exp1_model_col, exp1_test_col]].copy()
exp1_clean.columns = ["model", "test_accuracy"]


# ----- Exp2 -----
exp2_model_col = find_col(exp2, ["model", "backbone"])
exp2_test_col = find_col(exp2, ["test_accuracy", "test_acc", "test_acc_at_val_threshold"])

exp2_clean = exp2[[exp2_model_col, exp2_test_col]].copy()
exp2_clean.columns = ["model", "test_accuracy"]


# ----- Exp3 -----
exp3_test_col = find_col(exp3, ["test_accuracy", "test_acc"])

exp3_clean = pd.DataFrame([{
    "model": "ResNet18_Pretrained",
    "test_accuracy": exp3[exp3_test_col].iloc[0],
}])


# ----- Combine -----
df = pd.concat([exp1_clean, exp2_clean, exp3_clean], ignore_index=True)


# ----- AUC -----
if "model" not in auc.columns:
    auc = auc.rename(columns={"backbone": "model", "method": "model"})

df = df.merge(auc[["model", "auc"]], on="model", how="left")


# ----- One-shot -----
if "method" in oneshot1.columns:
    oneshot1 = oneshot1.rename(columns={"method": "model"})

if "backbone" in oneshot2.columns:
    oneshot2 = oneshot2.rename(columns={"backbone": "model"})

oneshot = pd.concat([oneshot1, oneshot2, exp3_oneshot], ignore_index=True)
df = df.merge(
    oneshot[["model", "oneshot_acc_N2", "oneshot_acc_N5", "oneshot_acc_N20"]],
    on="model",
    how="left"
)

df.to_csv("outputs/final_results.csv", index=False)

print("\nFinal table:")
print(df)