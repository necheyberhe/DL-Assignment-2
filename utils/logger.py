import pandas as pd
from pathlib import Path


def append_result(csv_path, row_dict):
    csv_path = Path(csv_path)

    row_df = pd.DataFrame([row_dict])

    if not csv_path.exists():
        row_df.to_csv(csv_path, index=False)
    else:
        row_df.to_csv(
            csv_path,
            mode="a",
            header=False,
            index=False
        )