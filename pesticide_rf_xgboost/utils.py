import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt


def get_xy(df):
    y = df["EPEST_HIGH_KG"].values

    X_cols = [col for col in df.columns if "item" in col.lower()]
    X = df[X_cols].values

    print(f"X: {X.shape}, y: {y.shape}")

    return X, y


def save_to_json(data, filename):
    """Save a Python object to a JSON file.

    Args:
        data: Python object to save (usually a dictionary or a list).
        filename (str): Path to the JSON file.

    Returns:
        None
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def round_dict(d):
    return {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in d.items()}


def get_metrics(y, y_pred, pesticide=None, model_type=None, save_folder="results"):
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    metrics_d = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    pprint(round_dict(metrics_d), indent=4)

    if pesticide and model_type:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{pesticide}_{model_type}_metrics.json")
        save_to_json(metrics_d, save_path)
    return metrics_d


def get_feat_importance_series(
    feat_importance,
    feat_dict,
    pesticide=None,
    model_type=None,
    tag=None,
    save_folder="results",
):
    all_rows = []
    for i, (feature_code, feature_name) in enumerate(feat_dict.items()):
        row = [feature_code, feature_name, feat_importance[i]]
        all_rows.append(row)
        if i == len(feat_importance) - 1:
            # break incase not all feat dict items are used
            # print("Not all feature_dict features were used")
            break

    feat_importance = pd.DataFrame(
        all_rows, columns=["feature_code", "feature_name", "importance"]
    )
    feat_importance = feat_importance.sort_values("importance", ascending=False)

    feat_importance.index.name = "feature"
    if pesticide and model_type and tag:
        save_path = os.path.join(
            save_folder, f"{pesticide}_{model_type}_feat_importance_{tag}.csv"
        )
        feat_importance.to_csv(save_path, index=False)
    return feat_importance


def make_fig(
    feat_importance, pesticide=None, model_type=None, tag=None, save_folder="results"
):
    feat_importance_series = feat_importance.set_index("feature_name")["importance"]
    ax = feat_importance_series.head(15).sort_values().plot(kind="barh")
    fig = ax.get_figure()

    if pesticide and model_type and tag:
        save_path = os.path.join(
            save_folder, f"{pesticide}_{model_type}_feat_importance_{tag}.png"
        )
        plt.savefig(
            save_path,
            bbox_inches="tight",
            facecolor="white",
        )
    # plt.show(block=False)
