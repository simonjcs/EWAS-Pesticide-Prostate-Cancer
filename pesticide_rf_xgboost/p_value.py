import numpy as np
from sklearn.inspection import permutation_importance
import pandas as pd
import os


def simulate_permute_p_values(
    clf,
    X_test,
    y_test,
    feat_importances,
    feat_dict,
    pesticide=None,
    model_type=None,
    total_iterations=100,
    n_jobs=-1,
    save_folder="results",
):
    """
    Simulate p-values by shuffling the outcome and comparing the permutation importance.
    """
    actual_importances = feat_importances.importances_mean[:, np.newaxis]
    importance_std = feat_importances.importances_std
    prev_importances = feat_importances.importances

    finished_n = prev_importances.shape[1]

    iterations_remaining = total_iterations - finished_n
    print(
        f"Simulating {iterations_remaining} more iterations for a total of {total_iterations}"
    )
    new_importances = permutation_importance(
        clf, X_test, y_test, n_repeats=iterations_remaining, n_jobs=n_jobs
    ).importances

    all_importances = np.concatenate((prev_importances, new_importances), axis=1)

    p_values = (all_importances >= actual_importances).mean(axis=1)

    p_values_df = get_p_value_df(
        p_values,
        feat_dict,
        importance_std,
        pesticide=pesticide,
        model_type=model_type,
        save_folder=save_folder,
    )

    return p_values_df


def get_p_value_df(
    p_values,
    feat_dict,
    importance_std,
    pesticide=None,
    model_type=None,
    save_folder="results",
):
    all_rows = []
    for i, (feature_code, feature_name) in enumerate(feat_dict.items()):
        row = [feature_code, feature_name, p_values[i], importance_std[i]]
        all_rows.append(row)
        if i == len(p_values) - 1:
            # break incase not all feat dict items are used
            break

    p_values_df = pd.DataFrame(
        all_rows, columns=["feature_code", "feature_name", "p-values", "importance_std"]
    )
    p_values_df = p_values_df.sort_values(by="p-values", ascending=False)

    if pesticide and model_type:
        save_path = os.path.join(save_folder, f"{pesticide}_{model_type}_p_value.csv")
        p_values_df.to_csv(save_path, index=False)
    return p_values_df
