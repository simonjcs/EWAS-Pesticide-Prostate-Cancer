import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from utils import get_feat_importance_series, get_metrics, get_xy, make_fig
from p_value import simulate_permute_p_values


def get_data(pesticide, outcome_df, feature_df, test_size=0.3):
    print(f"Getting data for {pesticide}...")
    outcome_for_pesticide_df = outcome_df[outcome_df["content"] == pesticide]
    outcome_for_pesticide_df = outcome_for_pesticide_df.join(feature_df, how="inner")

    train_df, test_df = train_test_split(outcome_for_pesticide_df, test_size=test_size)

    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)
    return X_train, y_train, X_test, y_test


def train_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    pesticide,
    model_type="rf",
    n_jobs=-1,
    save_folder="results",
):
    print(
        f"Training and evaluating {pesticide} with {model_type} and n_jobs={n_jobs}..."
    )
    if model_type == "rf":
        clf = RandomForestRegressor(n_jobs=n_jobs)
    elif model_type == "xgboost":
        clf = xgb.XGBRegressor(n_jobs=n_jobs)
    else:
        raise ValueError(
            "Invalid model_type. Choose either 'rf' for RandomForest or 'xgboost' for XGBoost."
        )

    clf = RandomForestRegressor(n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics_d = get_metrics(
        y_test,
        y_pred,
        pesticide=pesticide,
        model_type=model_type,
        save_folder=save_folder,
    )
    return clf, metrics_d


def get_mdi(clf, featcode2name, pesticide, model_type="rf", save_folder="results"):
    print(f"Getting MDI for {pesticide}...")

    feat_importance_mdi = clf.feature_importances_
    feat_importance_mdi = get_feat_importance_series(
        feat_importance_mdi,
        featcode2name,
        pesticide=pesticide,
        model_type=model_type,
        tag="mdi",
        save_folder=save_folder,
    )
    fig = make_fig(
        feat_importance_mdi,
        pesticide=pesticide,
        model_type=model_type,
        tag="mdi",
        save_folder=save_folder,
    )
    return feat_importance_mdi


def get_permute_importance(
    clf,
    X_test,
    y_test,
    featcode2name,
    pesticide,
    model_type="rf",
    n_jobs=-1,
    permute_n_repeats=5,
    p_value_total_iterations=100,
    save_folder="results",
):
    print(
        f"Getting permutation importance for {pesticide} with n_repeats={permute_n_repeats} and n_jobs={n_jobs}..."
    )

    permute_result = permutation_importance(
        clf, X_test, y_test, n_jobs=n_jobs, n_repeats=permute_n_repeats
    )

    feat_importance_permute = permute_result.importances_mean

    # You can print or return p-values here for further use
    # print("P-values:", p_values)

    tag = f"permute_{permute_n_repeats}"
    feat_importance_permute = get_feat_importance_series(
        feat_importance_permute,
        featcode2name,
        pesticide=pesticide,
        model_type=model_type,
        tag=tag,
        save_folder=save_folder,
    )
    fig = make_fig(
        feat_importance_permute,
        pesticide=pesticide,
        model_type=model_type,
        tag=tag,
        save_folder=save_folder,
    )

    # Simulate p-values using the permute_result
    if p_value_total_iterations:
        p_values = simulate_permute_p_values(
            clf,
            X_test,
            y_test,
            permute_result,
            featcode2name,
            pesticide=pesticide,
            model_type=model_type,
            total_iterations=p_value_total_iterations,
            n_jobs=n_jobs,
            save_folder=save_folder,
        )

    return feat_importance_permute
